/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

class NegTrainCw2vecOp : public OpKernel {
 public:
  explicit NegTrainCw2vecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    base_.Init(0, 0);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));

    std::vector<int32> vocab_count;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_count", &vocab_count));

    std::vector<float> vocab_weights;
    vocab_weights.reserve(vocab_count.size());
    for (const auto& f : vocab_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights.push_back(r);
    }
    sampler_ = new random::DistributionSampler(vocab_weights);
  }

  ~NegTrainCw2vecOp() { delete sampler_; }

  void Compute(OpKernelContext* ctx) override {
    // signed int32: 2,147,483,647
    // signed int64: 9,223,372,036,854,775,807

    // Wikipedia (several years back):
    // word_vocab_size: 299,529
    // stroke_vocab_size: 2,572,079

    // embeddings for stroke n-grams
    Tensor w_in = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()),
                errors::InvalidArgument("Must be a matrix"));

    // embeddings for context words
    Tensor w_out = ctx->mutable_input(1, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_out.shape()),
                errors::InvalidArgument("Must be a matrix"));

    // embedding dims must match
    OP_REQUIRES(ctx, w_in.dim_size(1) == w_out.dim_size(1),
                errors::InvalidArgument("w_in.shape[1] == w_out.shape[1]"));

    const Tensor& examples = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()),
                errors::InvalidArgument("Must be a vector"));

    const Tensor& labels = ctx->input(3);

    const Tensor& end_indices = ctx->input(4);
    OP_REQUIRES(ctx, end_indices.shape() == labels.shape(),
                errors::InvalidArgument("end_indices.shape == labels.shape"));

    const Tensor& learning_rate = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()),
                errors::InvalidArgument("Must be a scalar"));

    auto Tw_in = w_in.matrix<float>();
    auto Tw_out = w_out.matrix<float>();
    auto Texamples = examples.flat<int32>();
    auto Tlabels = labels.flat<int32>();
    auto Tend_indices = end_indices.flat<int32>();
    auto lr = learning_rate.scalar<float>()();

    const int32 stroke_vocab_size = w_in.dim_size(0);
    const int32 word_vocab_size = w_out.dim_size(0);
    const int32 dims = w_in.dim_size(1);
    const int32 batch_size = labels.dim_size(0);

    OP_REQUIRES(ctx, word_vocab_size == sampler_->num(),
                errors::InvalidArgument(
                  "word_vocab_size mismatches: ", word_vocab_size,
                  " vs. ", sampler_->num()));

    // the following loop needs 2 random 32-bit values per negative
    // sample. We reserve 8 values per sample just in case the
    // underlying implementation changes.
    auto rnd = base_.ReserveSamples32(batch_size * num_samples_ * 8);
    random::SimplePhilox srnd(&rnd);

    // update for q, for all q in S(w)
    Tensor q_update_buf(DT_FLOAT, TensorShape({dims}));
    auto Tq_update_acc = q_update_buf.flat<float>();

    // sum of q, for all q in S(w)
    Tensor qsum_buf(DT_FLOAT, TensorShape({dims}));
    auto Tqsum_acc = qsum_buf.flat<float>();

    // do we need to reset the buffer
    bool buf_good;

    // sigmoid related
    Tensor g_buf(DT_FLOAT, TensorShape({}));
    auto g = g_buf.scalar<float>();

    int32 begin_index; // begin index of the current center word
    int32 end_index;   // end index of the current center word
    float num_qs;      // number of stroke n-grams

    for (int32 i_example = 0; i_example < batch_size; ++i_example) {
      // get the current context word's embedding
      const int32 c_id = Tlabels(i_example);
      auto c = Tw_out.chip<0>(c_id);

      // begin and end index (into the examples) of w - the current center word
      end_index = Tend_indices(i_example);
      if (i_example == 0) {begin_index = 0;}
      else {begin_index = Tend_indices(i_example-1) + 1;}
      num_qs = end_index - begin_index + 1;

      // buf needs to be reseted for each example
      buf_good = false;

      // positive
      // calculate sum(q) for all q in S(w)
      for (int32 i_q = begin_index; i_q <= end_index; ++i_q) {
       // get q's embedding
        const int32 q_id = Texamples(i_q);
        auto q = Tw_in.chip<0>(q_id);

        if (not buf_good) {
         // clear the old buf
          Tqsum_acc = q;
          buf_good = true;
        } else {
          Tqsum_acc += q;
        }
      }

      // 1 / (1 + e^x) where x = sim(w, c) / k
      g = (1.f + ((Tqsum_acc * c).sum() / num_qs).exp()).inverse();

      Tq_update_acc = (lr * g() / num_qs) * c;

      // back prop into the true label
      c += (lr * g() / num_qs) * Tqsum_acc;

      // negative
      for (int i_false = 0; i_false < num_samples_; ++i_false) {
        // get the embedding of c'
        const int c_prime_id = sampler_->Sample(&srnd);
        if (c_prime_id == c_id) continue; // skip
        auto c_prime = Tw_out.chip<0>(c_prime_id);

        // -sigmoid(x) where x = sim(w, c') / k
        g = -(    ( 1.f + (-(Tqsum_acc * c_prime).sum() / num_qs).exp() ).inverse()    );

        Tq_update_acc += (lr * g() / num_qs) * c_prime;

        // back prop into c_prime
        c_prime += (lr * g() / num_qs) * Tqsum_acc;
      }

      // back prop into q for all q in S(w)
      for (int32 i_q = begin_index; i_q <= end_index; ++i_q) {
        const int32 q_id = Texamples(i_q);
        Tw_in.chip<0>(q_id) += Tq_update_acc;
      }
    }
  }

 private:
  int32 num_samples_ = 0;
  random::DistributionSampler* sampler_ = nullptr;
  GuardedPhiloxRandom base_;
};

REGISTER_KERNEL_BUILDER(Name("NegTrainCw2vec").Device(DEVICE_CPU), NegTrainCw2vecOp);

}
