// ME3 (the cleanup): the expert cache's host service (service_loop.h).

#include "fn/runtime/service_loop.h"

namespace fn::runtime::svc {

void loop::start(work * w, refill * r, int32_t layers) {
    w_ = w;
    r_ = r;
    posted_.assign(size_t(layers), 0);
    done_.assign(size_t(layers), 0);
    th_ = std::thread(&loop::main, this);
}

void loop::post(int32_t layer, int32_t rows, int32_t par) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        const uint32_t seq = ++seq_;
        queue_.push_back({layer, rows, seq, par, false});
        posted_[size_t(layer)] = seq;
        ++queued_;
    }
    cv_.notify_one();
}

void loop::post_pass(int32_t layers, int32_t rows) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t l = 0; l < layers; ++l) {
            const uint32_t seq = ++seq_;
            queue_.push_back({l, rows, seq, 0, true});
            posted_[size_t(l)] = seq;
            ++queued_;
        }
    }
    cv_.notify_one();
}

void loop::post_failed(int32_t layer, int32_t rows, int32_t par, const std::string & err) {
    record_failure(err);
    post(layer, rows, par);
}

void loop::wait_idle() {
    ++idle_waiters_; // a refill batch being submitted ends after its current move
    {
        std::unique_lock<std::mutex> lk(mu_);
        idle_cv_.wait(lk, [&]() { return queue_.empty() && !busy_; });
    }
    --idle_waiters_;
}

void loop::wait_layer(int32_t layer) {
    std::unique_lock<std::mutex> lk(mu_);
    idle_cv_.wait(lk, [&]() { return done_[size_t(layer)] == posted_[size_t(layer)]; });
}

// the loop's lock taken and released before the notification: a service evaluating ready() under it either sees the
// producer's change or is already waiting
void loop::notify() {
    {
        std::lock_guard<std::mutex> lk(mu_);
    }
    cv_.notify_one();
}

void loop::abandon(const std::string & why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        abandon_ = "the replay's launch failed: " + why;
    }
    quit_ = true;
    cv_.notify_all();
}

void loop::release(const std::string & why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        abandon_ = why;
    }
    record_failure(why);
    quit_ = true;
    cv_.notify_all();
}

void loop::stop() {
    if (!th_.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    quit_ = true; // a replay's window whose stamp never comes is unblocked
    cv_.notify_all();
    th_.join(); // after every queued job: no stream is left waiting on a flag
}

std::string loop::error() {
    std::lock_guard<std::mutex> lk(mu_);
    return fail_err_;
}

void loop::record_failure(const std::string & e) {
    bool first = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!failed_) {
            fail_err_ = e;
            failed_ = true;
            first = true;
        }
    }
    if (first && r_ != nullptr) {
        r_->failed(e);
    }
}

// each queued job in order; with none queued (and no failure), the refill's next batch
void loop::main() {
    for (;;) {
        job j;
        bool rf = false;
        size_t a = 0, b = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&]() { return stop_ || !queue_.empty() || (!failed_ && r_ != nullptr && r_->ready()); });
            if (!queue_.empty()) {
                j = queue_.front();
                queue_.pop_front();
                --queued_;
            } else if (!stop_) {
                r_->take(a, b);
                rf = true;
            } else {
                return; // stopping, every job ended
            }
            busy_ = true;
        }
        if (rf) {
            std::string e;
            if (r_->submit(a, b, [this]() { return yield(); }, e)) {
                ++batches_;
            } else {
                record_failure(e);
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                busy_ = false;
            }
            idle_cv_.notify_all();
            continue;
        }
        std::string e;
        const bool came = w_->await(j, quit_, e);
        if (!came && e.empty()) {
            std::lock_guard<std::mutex> lk(mu_);
            e = "expert cache: a replay's window not run: " +
                (abandon_.empty() ? std::string("the cache released") : abandon_);
        }
        if (came && !failed_ && w_->serve(j, e)) {
            ++served_;
        } else {
            if (!came || !failed_) {
                record_failure(e);
            }
            w_->unblock(j); // after the descriptor (the resolve's flag reset), or at once when it never comes
            ++unblocked_;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_[size_t(j.layer)] = j.seq;
            busy_ = false;
        }
        idle_cv_.notify_all();
    }
}

} // namespace fn::runtime::svc
