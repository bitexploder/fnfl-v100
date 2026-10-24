#include "fn/runtime/json_check.h"

#include <cctype>

namespace fn::runtime {

namespace {

struct parser {
    const std::string & s;
    size_t i = 0;
    int depth = 0;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
            ++i;
        }
    }
    bool lit(const char * w) {
        size_t k = 0;
        for (; w[k] != '\0'; ++k) {
            if (i + k >= s.size() || s[i + k] != w[k]) {
                return false;
            }
        }
        i += k;
        return true;
    }
    bool digits() {
        const size_t a = i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])) != 0) {
            ++i;
        }
        return i > a;
    }
    bool number() {
        if (i < s.size() && s[i] == '-') {
            ++i;
        }
        if (i < s.size() && s[i] == '0') {
            ++i;
        } else if (!digits()) {
            return false;
        }
        if (i < s.size() && s[i] == '.') {
            ++i;
            if (!digits()) {
                return false;
            }
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
                ++i;
            }
            if (!digits()) {
                return false;
            }
        }
        return true;
    }
    bool string() {
        if (i >= s.size() || s[i] != '"') {
            return false;
        }
        for (++i; i < s.size(); ++i) {
            const unsigned char ch = static_cast<unsigned char>(s[i]);
            if (ch == '"') {
                ++i;
                return true;
            }
            if (ch < 0x20) {
                return false;
            }
            if (ch == '\\') {
                if (++i >= s.size()) {
                    return false;
                }
                const char e = s[i];
                if (e == 'u') {
                    for (int k = 0; k < 4; ++k) {
                        if (++i >= s.size() || std::isxdigit(static_cast<unsigned char>(s[i])) == 0) {
                            return false;
                        }
                    }
                } else if (e != '"' && e != '\\' && e != '/' && e != 'b' && e != 'f' && e != 'n' && e != 'r' &&
                           e != 't') {
                    return false;
                }
            }
        }
        return false;
    }
    bool value() {
        ws();
        if (i >= s.size() || ++depth > 64) {
            return false;
        }
        bool ok = false;
        const char ch = s[i];
        if (ch == '{') {
            ++i;
            ws();
            ok = true;
            if (i < s.size() && s[i] == '}') {
                ++i;
            } else {
                for (;;) {
                    ws();
                    if (!string()) {
                        ok = false;
                        break;
                    }
                    ws();
                    if (i >= s.size() || s[i] != ':') {
                        ok = false;
                        break;
                    }
                    ++i;
                    if (!value()) {
                        ok = false;
                        break;
                    }
                    ws();
                    if (i < s.size() && s[i] == ',') {
                        ++i;
                        continue;
                    }
                    ok = i < s.size() && s[i] == '}';
                    i += ok ? 1 : 0;
                    break;
                }
            }
        } else if (ch == '[') {
            ++i;
            ws();
            ok = true;
            if (i < s.size() && s[i] == ']') {
                ++i;
            } else {
                for (;;) {
                    if (!value()) {
                        ok = false;
                        break;
                    }
                    ws();
                    if (i < s.size() && s[i] == ',') {
                        ++i;
                        continue;
                    }
                    ok = i < s.size() && s[i] == ']';
                    i += ok ? 1 : 0;
                    break;
                }
            }
        } else if (ch == '"') {
            ok = string();
        } else if (ch == 't') {
            ok = lit("true");
        } else if (ch == 'f') {
            ok = lit("false");
        } else if (ch == 'n') {
            ok = lit("null");
        } else {
            ok = number();
        }
        --depth;
        return ok;
    }
};

} // namespace

bool json_valid(const std::string & s, size_t & at) {
    parser p{s};
    const bool ok = p.value();
    p.ws();
    at = p.i;
    return ok && p.i == s.size();
}

} // namespace fn::runtime
