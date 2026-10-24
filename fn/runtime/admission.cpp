#include "fn/runtime/admission.h"

#include <algorithm>

namespace fn::runtime::adm {

int32_t table::held(int32_t layer) const {
    int32_t n = 0;
    for (int32_t i = layer * per_layer; i < (layer + 1) * per_layer; ++i) {
        n += r_[size_t(i)].st != state::none;
    }
    return n;
}

int32_t table::held_all() const {
    int32_t n = 0;
    for (const rec & x : r_) {
        n += x.st != state::none;
    }
    return n;
}

int32_t table::in_state(state s) const {
    int32_t n = 0;
    for (const rec & x : r_) {
        n += x.st == s;
    }
    return n;
}

int32_t table::of_slot(int32_t s) const {
    for (int32_t i = 0; i < max_recs; ++i) {
        if (r_[size_t(i)].st != state::none && r_[size_t(i)].slot == s) {
            return i;
        }
    }
    return -1;
}

bool table::in_flight(int32_t key) const {
    for (const rec & x : r_) {
        if ((x.st == state::copying || x.st == state::landed) && x.key == key) {
            return true;
        }
    }
    return false;
}

bool table::reserve(int32_t layer, int32_t s) {
    if (layer < 0 || layer >= n_layers || s < 0 || of_slot(s) >= 0) {
        return false;
    }
    for (int32_t i = layer * per_layer; i < (layer + 1) * per_layer; ++i) {
        rec & x = r_[size_t(i)];
        if (x.st == state::none) {
            x = rec{s, layer, -1, gen_++, state::empty};
            return true;
        }
    }
    return false;
}

bool table::start(int32_t i, int32_t key) {
    rec & x = r_[size_t(i)];
    if (x.st != state::empty || key < 0 || key / 512 != x.layer || in_flight(key)) {
        return false;
    }
    x.key = key;
    x.st = state::copying;
    return true;
}

bool table::land(int32_t i) {
    rec & x = r_[size_t(i)];
    if (x.st != state::copying) {
        return false;
    }
    x.st = state::landed;
    return true;
}

bool table::publish(int32_t i, int32_t & slot, int32_t & key) {
    rec & x = r_[size_t(i)];
    if (x.st != state::landed) {
        return false;
    }
    slot = x.slot;
    key = x.key;
    x = rec();
    return true;
}

bool table::invalidate(int32_t i) {
    rec & x = r_[size_t(i)];
    if (x.st != state::copying && x.st != state::landed) {
        return false;
    }
    x.key = -1;
    x.gen = gen_++;
    x.st = state::empty;
    return true;
}

std::vector<int32_t> table::discard_all() {
    std::vector<int32_t> slots;
    for (rec & x : r_) {
        if (x.st != state::none) {
            slots.push_back(x.slot);
        }
        x = rec();
    }
    ++gen_;
    return slots;
}

void table::return_from_loan() {
    for (rec & x : r_) {
        if (x.st != state::none) {
            x.key = -1;
            x.gen = gen_++;
            x.st = state::empty;
        }
    }
}

std::vector<cand> victims(std::vector<cand> c, int32_t need) {
    std::sort(c.begin(), c.end(), [](const cand & a, const cand & b) {
        return a.usage != b.usage ? a.usage < b.usage : a.key != b.key ? a.key < b.key : a.slot < b.slot;
    });
    c.resize(size_t(std::clamp<int32_t>(need, 0, int32_t(c.size()))));
    return c;
}

std::vector<miss> admit(std::vector<miss> m, int32_t n_avail) {
    std::stable_sort(m.begin(), m.end(), [](const miss & a, const miss & b) {
        return a.rows != b.rows ? a.rows > b.rows : a.order < b.order;
    });
    m.resize(size_t(std::clamp<int32_t>(n_avail, 0, int32_t(m.size()))));
    return m;
}

} // namespace fn::runtime::adm
