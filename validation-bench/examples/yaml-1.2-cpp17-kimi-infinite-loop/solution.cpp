#include <bits/stdc++.h>
using namespace std;

vector<char32_t> decode(const vector<uint8_t>& b) {
    vector<char32_t> out;
    size_t i = 0;
    auto get = [&](size_t idx)->uint8_t { return idx < b.size() ? b[idx] : 0; };
    if (b.size() >= 4 && get(0)==0x00 && get(1)==0x00 && get(2)==0xFE && get(3)==0xFF) {
        i = 4;
        while (i+3 < b.size()) {
            char32_t cp = ((char32_t)b[i]<<24) | ((char32_t)b[i+1]<<16) | ((char32_t)b[i+2]<<8) | b[i+3];
            out.push_back(cp); i += 4;
        }
    } else if (b.size() >= 4 && get(0)==0x00 && get(1)==0x00 && get(2)==0x00 && get(3)<=0x7F) {
        while (i+3 < b.size()) {
            char32_t cp = ((char32_t)b[i]<<24) | ((char32_t)b[i+1]<<16) | ((char32_t)b[i+2]<<8) | b[i+3];
            out.push_back(cp); i += 4;
        }
    } else if (b.size() >= 4 && get(0)==0xFF && get(1)==0xFE && get(2)==0x00 && get(3)==0x00) {
        i = 4;
        while (i+3 < b.size()) {
            char32_t cp = ((char32_t)b[i+3]<<24) | ((char32_t)b[i+2]<<16) | ((char32_t)b[i+1]<<8) | b[i];
            out.push_back(cp); i += 4;
        }
    } else if (b.size() >= 4 && get(1)==0x00 && get(2)==0x00 && get(3)==0x00 && get(0)<=0x7F) {
        while (i+3 < b.size()) {
            char32_t cp = ((char32_t)b[i+3]<<24) | ((char32_t)b[i+2]<<16) | ((char32_t)b[i+1]<<8) | b[i];
            out.push_back(cp); i += 4;
        }
    } else if (b.size() >= 2 && get(0)==0xFE && get(1)==0xFF) {
        i = 2;
        while (i+1 < b.size()) {
            char32_t w = ((char32_t)b[i]<<8) | b[i+1];
            if (w >= 0xD800 && w <= 0xDBFF && i+3 < b.size()) {
                char32_t w2 = ((char32_t)b[i+2]<<8) | b[i+3];
                if (w2 >= 0xDC00 && w2 <= 0xDFFF) {
                    out.push_back(0x10000 + ((w&0x3FF)<<10) + (w2&0x3FF));
                    i += 4; continue;
                }
            }
            out.push_back(w); i += 2;
        }
    } else if (b.size() >= 2 && get(0)==0x00 && get(1)<=0x7F) {
        while (i+1 < b.size()) {
            char32_t w = ((char32_t)b[i]<<8) | b[i+1];
            if (w >= 0xD800 && w <= 0xDBFF && i+3 < b.size()) {
                char32_t w2 = ((char32_t)b[i+2]<<8) | b[i+3];
                if (w2 >= 0xDC00 && w2 <= 0xDFFF) {
                    out.push_back(0x10000 + ((w&0x3FF)<<10) + (w2&0x3FF));
                    i += 4; continue;
                }
            }
            out.push_back(w); i += 2;
        }
    } else if (b.size() >= 2 && get(0)==0xFF && get(1)==0xFE) {
        i = 2;
        while (i+1 < b.size()) {
            char32_t w = b[i] | ((char32_t)b[i+1]<<8);
            if (w >= 0xD800 && w <= 0xDBFF && i+3 < b.size()) {
                char32_t w2 = b[i+2] | ((char32_t)b[i+3]<<8);
                if (w2 >= 0xDC00 && w2 <= 0xDFFF) {
                    out.push_back(0x10000 + ((w&0x3FF)<<10) + (w2&0x3FF));
                    i += 4; continue;
                }
            }
            out.push_back(w); i += 2;
        }
    } else if (b.size() >= 2 && get(0)<=0x7F && get(1)==0x00) {
        while (i+1 < b.size()) {
            char32_t w = b[i] | ((char32_t)b[i+1]<<8);
            if (w >= 0xD800 && w <= 0xDBFF && i+3 < b.size()) {
                char32_t w2 = b[i+2] | ((char32_t)b[i+3]<<8);
                if (w2 >= 0xDC00 && w2 <= 0xDFFF) {
                    out.push_back(0x10000 + ((w&0x3FF)<<10) + (w2&0x3FF));
                    i += 4; continue;
                }
            }
            out.push_back(w); i += 2;
        }
    } else {
        if (b.size() >= 3 && get(0)==0xEF && get(1)==0xBB && get(2)==0xBF) i = 3;
        while (i < b.size()) {
            uint8_t c1 = b[i];
            if (c1 < 0x80) { out.push_back(c1); ++i; continue; }
            if ((c1 & 0xE0) == 0xC0 && i+1 < b.size()) {
                uint8_t c2 = b[i+1];
                if ((c2 & 0xC0) == 0x80) {
                    char32_t cp = ((c1 & 0x1F) << 6) | (c2 & 0x3F);
                    if (cp >= 0x80) { out.push_back(cp); i += 2; continue; }
                }
            } else if ((c1 & 0xF0) == 0xE0 && i+2 < b.size()) {
                uint8_t c2 = b[i+1], c3 = b[i+2];
                if ((c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {
                    char32_t cp = ((c1 & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
                    if (cp >= 0x800) { out.push_back(cp); i += 3; continue; }
                }
            } else if ((c1 & 0xF8) == 0xF0 && i+3 < b.size()) {
                uint8_t c2 = b[i+1], c3 = b[i+2], c4 = b[i+3];
                if ((c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80 && (c4 & 0xC0) == 0x80) {
                    char32_t cp = ((c1 & 0x07) << 18) | ((c2 & 0x3F) << 12) | ((c3 & 0x3F) << 6) | (c4 & 0x3F);
                    if (cp >= 0x10000 && cp <= 0x10FFFF) { out.push_back(cp); i += 4; continue; }
                }
            }
            out.push_back(c1); ++i;
        }
    }
    return out;
}

enum Context { BLOCK_OUT, BLOCK_IN, BLOCK_KEY, FLOW_OUT, FLOW_IN, FLOW_KEY };
enum Chomp { STRIP, CLIP, KEEP };

struct Parser {
    vector<char32_t> s;
    size_t pos = 0;
    Parser(vector<char32_t> str) : s(move(str)) {}

    bool eof() const { return pos >= s.size(); }
    char32_t peek() const { return eof() ? 0 : s[pos]; }
    bool at(char32_t c) const { return peek() == c; }
    bool eat(char32_t c) { if (at(c)) { ++pos; return true; } return false; }
    bool match_str(const char* p) {
        size_t saved = pos;
        for (; *p; ++p) if (!eat((char32_t)(unsigned char)*p)) { pos = saved; return false; }
        return true;
    }

    bool is_start_of_line() const {
        size_t p = pos;
        while (p > 0 && s[p-1] == 0xFEFF) --p;
        if (p == 0) return true;
        char32_t c = s[p-1];
        return c == '\n' || c == '\r';
    }

    bool is_printable(char32_t c) const {
        if (c == 0x09 || c == 0x0A || c == 0x0D) return true;
        if (c >= 0x20 && c <= 0x7E) return true;
        if (c == 0x85) return true;
        if (c >= 0xA0 && c <= 0xD7FF) return true;
        if (c >= 0xE000 && c <= 0xFFFD) return true;
        if (c >= 0x10000 && c <= 0x10FFFF) return true;
        return false;
    }
    bool is_nb_json(char32_t c) const {
        if (c == 0x09) return true;
        if (c >= 0x20 && c <= 0x10FFFF) return true;
        return false;
    }
    bool is_nb_char(char32_t c) const {
        return is_printable(c) && c != '\n' && c != '\r' && c != 0xFEFF;
    }
    bool is_ns_char(char32_t c) const {
        return is_nb_char(c) && c != ' ' && c != '\t';
    }
    bool is_whitespace(char32_t c) const { return c == ' ' || c == '\t'; }
    bool is_dec_digit(char32_t c) const { return c >= '0' && c <= '9'; }
    bool is_hex_digit(char32_t c) const {
        return is_dec_digit(c) || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
    }
    bool is_ascii_letter(char32_t c) const {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }
    bool is_word_char(char32_t c) const {
        return is_dec_digit(c) || is_ascii_letter(c) || c == '-';
    }
    bool is_flow_indicator(char32_t c) const {
        return c == ',' || c == '[' || c == ']' || c == '{' || c == '}';
    }
    bool is_indicator(char32_t c) const {
        return c == '-' || c == '?' || c == ':' || c == ',' || c == '[' || c == ']' || c == '{' || c == '}' || c == '#' || c == '&' || c == '*' || c == '!' || c == '|' || c == '>' || c == '\'' || c == '"' || c == '%' || c == '@' || c == '`';
    }
    bool is_uri_char(char32_t c) const {
        if (is_word_char(c)) return true;
        static const string sp = "#;/?:@&=+$,.!~*'()[]";
        return sp.find((char)c) != string::npos;
    }
    bool is_tag_char(char32_t c) const {
        if (c == '!' || is_flow_indicator(c)) return false;
        if (is_word_char(c)) return true;
        static const string sp = "#;/?:@&=+$_.!~*'()";
        return sp.find((char)c) != string::npos;
    }
    bool is_ns_anchor_char(char32_t c) const {
        return is_ns_char(c) && !is_flow_indicator(c);
    }

    bool match_break() {
        if (pos + 1 < s.size() && s[pos] == '\r' && s[pos+1] == '\n') { pos += 2; return true; }
        if (at('\r') || at('\n')) { ++pos; return true; }
        return false;
    }

    bool match_indent(int n) {
        if (n <= 0) return true;
        size_t saved = pos;
        for (int i = 0; i < n; ++i) if (!eat(' ')) { pos = saved; return false; }
        return true;
    }
    bool match_indent_less_than(int n) {
        if (n <= 0) return true;
        int k = 0;
        while (k < n && eat(' ')) ++k;
        return true;
    }
    bool match_indent_less_or_equal(int n) {
        if (n < 0) n = 0;
        int k = 0;
        while (k < n && eat(' ')) ++k;
        return true;
    }
    bool match_s_flow_line_prefix(int n) { return match_indent(n); }
    bool match_s_block_line_prefix(int n) { return match_indent(n); }
    bool match_s_line_prefix(int n, Context c) {
        if (c == BLOCK_OUT || c == BLOCK_IN || c == BLOCK_KEY) return match_s_block_line_prefix(n);
        return match_s_flow_line_prefix(n);
    }
    bool match_s_separate_in_line() {
        if (is_start_of_line()) return true;
        if (!is_whitespace(peek())) return false;
        while (is_whitespace(peek())) ++pos;
        return true;
    }

    bool parse_c_nb_comment_text() {
        if (!eat('#')) return false;
        while (!eof() && is_nb_char(peek())) ++pos;
        return true;
    }
    bool parse_b_comment() {
        return match_break() || eof();
    }
    bool parse_s_b_comment() {
        size_t saved = pos;
        if (match_s_separate_in_line()) parse_c_nb_comment_text();
        if (parse_b_comment()) return true;
        pos = saved;
        return false;
    }
    bool parse_l_comment() {
        size_t saved = pos;
        if (!match_s_separate_in_line()) { pos = saved; return false; }
        parse_c_nb_comment_text();
        if (parse_b_comment()) return true;
        pos = saved;
        return false;
    }
    bool parse_s_l_comments() {
        if (is_start_of_line()) {
            // ok
        } else {
            size_t saved = pos;
            if (!parse_s_b_comment()) { pos = saved; return false; }
        }
        while (true) {
            size_t saved = pos;
            if (parse_l_comment()) continue;
            pos = saved;
            break;
        }
        return true;
    }

    bool parse_s_separate_lines(int n, bool greedy) {
        size_t saved = pos;
        if (parse_s_l_comments()) {
            if (match_s_flow_line_prefix(n)) {
                if (greedy) while (eat(' ')) {}
                return true;
            }
        }
        pos = saved;
        if (match_s_separate_in_line()) return true;
        return false;
    }
    bool parse_s_separate(int n, Context c) {
        bool greedy = (c == FLOW_OUT || c == FLOW_IN || c == FLOW_KEY);
        if (c == BLOCK_KEY || c == FLOW_KEY) return match_s_separate_in_line();
        return parse_s_separate_lines(n, greedy);
    }

    bool parse_ns_directive_name() {
        if (eof() || !is_ns_char(peek())) return false;
        while (!eof() && is_ns_char(peek())) ++pos;
        return true;
    }
    bool parse_ns_directive_parameter() { return parse_ns_directive_name(); }
    bool parse_ns_reserved_directive() {
        if (!parse_ns_directive_name()) return false;
        while (true) {
            size_t saved = pos;
            if (match_s_separate_in_line() && parse_ns_directive_parameter()) continue;
            pos = saved;
            break;
        }
        return true;
    }
    bool parse_ns_yaml_version() {
        if (!is_dec_digit(peek())) return false;
        while (is_dec_digit(peek())) ++pos;
        if (!eat('.')) return false;
        if (!is_dec_digit(peek())) return false;
        while (is_dec_digit(peek())) ++pos;
        return true;
    }
    bool parse_ns_yaml_directive() {
        if (!match_str("YAML")) return false;
        if (!match_s_separate_in_line()) return false;
        return parse_ns_yaml_version();
    }
    bool parse_c_tag_handle() {
        size_t saved = pos;
        if (!eat('!')) return false;
        if (eat('!')) return true;
        if (is_word_char(peek())) {
            while (is_word_char(peek())) ++pos;
            if (eat('!')) return true;
            pos = saved;
            return false;
        }
        return true;
    }
    bool match_uri_char() {
        if (eat('%')) {
            if (pos + 1 < s.size() && is_hex_digit(s[pos]) && is_hex_digit(s[pos+1])) {
                pos += 2; return true;
            }
            return false;
        }
        char32_t c = peek();
        if (is_uri_char(c)) { ++pos; return true; }
        return false;
    }
    bool match_tag_char() {
        if (eat('%')) {
            if (pos + 1 < s.size() && is_hex_digit(s[pos]) && is_hex_digit(s[pos+1])) {
                pos += 2; return true;
            }
            return false;
        }
        char32_t c = peek();
        if (is_tag_char(c)) { ++pos; return true; }
        return false;
    }
    bool parse_ns_tag_prefix() {
        if (eat('!')) {
            while (true) { size_t saved = pos; if (match_uri_char()) continue; pos = saved; break; }
            return true;
        }
        if (!match_uri_char()) return false;
        while (true) { size_t saved = pos; if (match_uri_char()) continue; pos = saved; break; }
        return true;
    }
    bool parse_ns_tag_directive() {
        if (!match_str("TAG")) return false;
        if (!match_s_separate_in_line()) return false;
        if (!parse_c_tag_handle()) return false;
        if (!match_s_separate_in_line()) return false;
        return parse_ns_tag_prefix();
    }
    bool parse_l_directive() {
        if (!eat('%')) return false;
        size_t saved = pos;
        if (parse_ns_yaml_directive() || parse_ns_tag_directive() || parse_ns_reserved_directive()) {
            if (parse_s_l_comments()) return true;
        }
        pos = saved;
        return false;
    }

    bool parse_c_verbatim_tag() {
        if (!match_str("!<")) return false;
        if (!match_uri_char()) return false;
        while (true) { size_t saved = pos; if (match_uri_char()) continue; pos = saved; break; }
        return eat('>');
    }
    bool parse_c_ns_shorthand_tag() {
        if (!parse_c_tag_handle()) return false;
        if (!match_tag_char()) return false;
        while (true) { size_t saved = pos; if (match_tag_char()) continue; pos = saved; break; }
        return true;
    }
    bool parse_c_non_specific_tag() { return eat('!'); }
    bool parse_c_ns_tag_property() {
        return parse_c_verbatim_tag() || parse_c_ns_shorthand_tag() || parse_c_non_specific_tag();
    }
    bool parse_ns_anchor_name() {
        if (eof() || !is_ns_anchor_char(peek())) return false;
        while (!eof() && is_ns_anchor_char(peek())) ++pos;
        return true;
    }
    bool parse_c_ns_anchor_property() {
        return eat('&') && parse_ns_anchor_name();
    }
    bool parse_c_ns_properties(int n, Context c) {
        size_t saved = pos;
        if (parse_c_ns_tag_property()) {
            size_t saved2 = pos;
            if (parse_s_separate(n, c) && parse_c_ns_anchor_property()) return true;
            pos = saved2;
            return true;
        }
        pos = saved;
        if (parse_c_ns_anchor_property()) {
            size_t saved2 = pos;
            if (parse_s_separate(n, c) && parse_c_ns_tag_property()) return true;
            pos = saved2;
            return true;
        }
        return false;
    }
    bool parse_c_ns_alias_node() {
        return eat('*') && parse_ns_anchor_name();
    }
    bool parse_e_scalar() { return true; }
    bool parse_e_node() { return true; }

    bool parse_c_ns_esc_char() {
        if (!eat('\\')) return false;
        char32_t c = peek();
        switch (c) {
            case '0': case 'a': case 'b': case 't': case 'n': case 'v': case 'f': case 'r': case 'e':
            case ' ': case '"': case '/': case '\\': case 'N': case '_': case 'L': case 'P':
                ++pos; return true;
            case 'x': {
                ++pos;
                if (pos + 1 <= s.size() && is_hex_digit(s[pos]) && is_hex_digit(s[pos+1])) { pos += 2; return true; }
                return false;
            }
            case 'u': {
                ++pos;
                if (pos + 3 < s.size() && is_hex_digit(s[pos]) && is_hex_digit(s[pos+1]) && is_hex_digit(s[pos+2]) && is_hex_digit(s[pos+3])) { pos += 4; return true; }
                return false;
            }
            case 'U': {
                ++pos;
                if (pos + 7 < s.size()) {
                    bool ok = true;
                    for (int i = 0; i < 8; ++i) if (!is_hex_digit(s[pos+i])) ok = false;
                    if (ok) { pos += 8; return true; }
                }
                return false;
            }
            case '\t':
                ++pos; return true;
            default: return false;
        }
    }
    bool parse_nb_double_char() {
        if (parse_c_ns_esc_char()) return true;
        char32_t c = peek();
        if (c == '\\' || c == '"') return false;
        if (!is_nb_json(c)) return false;
        ++pos;
        return true;
    }
    bool is_ns_double_char(char32_t c) const {
        if (is_whitespace(c)) return false;
        if (c == '\\' || c == '"') return false;
        return is_nb_json(c);
    }
    bool parse_nb_double_one_line() {
        while (true) { size_t saved = pos; if (parse_nb_double_char()) continue; pos = saved; break; }
        return true;
    }
    bool parse_nb_ns_double_in_line() {
        while (true) {
            size_t saved = pos;
            while (is_whitespace(peek())) ++pos;
            if (is_ns_double_char(peek())) { ++pos; continue; }
            pos = saved;
            break;
        }
        return true;
    }
    bool parse_s_double_escaped(int n) {
        size_t saved = pos;
        while (is_whitespace(peek())) ++pos;
        if (!eat('\\')) { pos = saved; return false; }
        if (!match_break()) { pos = saved; return false; }
        while (parse_l_empty(n, FLOW_IN)) {}
        if (!match_s_flow_line_prefix(n)) { pos = saved; return false; }
        return true;
    }
    bool parse_s_double_break(int n) {
        return parse_s_double_escaped(n) || parse_s_flow_folded(n);
    }
    bool parse_s_flow_folded(int n) {
        if (!is_start_of_line() && is_whitespace(peek())) {
            while (is_whitespace(peek())) ++pos;
        }
        if (!parse_b_l_folded(n, FLOW_IN)) return false;
        if (!match_s_flow_line_prefix(n)) return false;
        while (eat(' ')) {}
        return true;
    }
    bool parse_s_double_next_line(int n) {
        if (!parse_s_double_break(n)) return false;
        size_t saved = pos;
        if (is_ns_double_char(peek())) {
            ++pos;
            parse_nb_ns_double_in_line();
            size_t saved2 = pos;
            if (parse_s_double_next_line(n)) return true;
            pos = saved2;
            while (is_whitespace(peek())) ++pos;
            return true;
        }
        pos = saved;
        return true;
    }
    bool parse_nb_double_multi_line(int n) {
        parse_nb_ns_double_in_line();
        size_t saved = pos;
        if (parse_s_double_next_line(n)) return true;
        pos = saved;
        while (is_whitespace(peek())) ++pos;
        return true;
    }
    bool parse_c_double_quoted(int n, Context c) {
        size_t saved = pos;
        if (!eat('"')) return false;
        bool ok;
        if (c == BLOCK_KEY || c == FLOW_KEY) ok = parse_nb_double_one_line();
        else ok = parse_nb_double_multi_line(n);
        if (!ok || !eat('"')) { pos = saved; return false; }
        return true;
    }

    bool parse_nb_single_char() {
        if (match_str("''")) return true;
        char32_t c = peek();
        if (c == '\'') return false;
        if (!is_nb_json(c)) return false;
        ++pos;
        return true;
    }
    bool is_ns_single_char(char32_t c) const {
        return !is_whitespace(c) && c != '\'' && is_nb_json(c);
    }
    bool parse_nb_single_one_line() {
        while (true) { size_t saved = pos; if (parse_nb_single_char()) continue; pos = saved; break; }
        return true;
    }
    bool parse_nb_ns_single_in_line() {
        while (true) {
            size_t saved = pos;
            while (is_whitespace(peek())) ++pos;
            if (is_ns_single_char(peek())) { ++pos; continue; }
            pos = saved;
            break;
        }
        return true;
    }
    bool parse_s_single_next_line(int n) {
        if (!parse_s_flow_folded(n)) return false;
        size_t saved = pos;
        if (is_ns_single_char(peek())) {
            ++pos;
            parse_nb_ns_single_in_line();
            size_t saved2 = pos;
            if (parse_s_single_next_line(n)) return true;
            pos = saved2;
            while (is_whitespace(peek())) ++pos;
            return true;
        }
        pos = saved;
        return true;
    }
    bool parse_nb_single_multi_line(int n) {
        parse_nb_ns_single_in_line();
        size_t saved = pos;
        if (parse_s_single_next_line(n)) return true;
        pos = saved;
        while (is_whitespace(peek())) ++pos;
        return true;
    }
    bool parse_c_single_quoted(int n, Context c) {
        size_t saved = pos;
        if (!eat('\'')) return false;
        bool ok;
        if (c == BLOCK_KEY || c == FLOW_KEY) ok = parse_nb_single_one_line();
        else ok = parse_nb_single_multi_line(n);
        if (!ok || !eat('\'')) { pos = saved; return false; }
        return true;
    }

    bool is_ns_plain_safe(char32_t c, Context ctx) const {
        if (!is_ns_char(c)) return false;
        if (ctx == FLOW_IN || ctx == FLOW_KEY) {
            if (is_flow_indicator(c)) return false;
        }
        return true;
    }
    bool parse_ns_plain_first(Context c) {
        char32_t ch = peek();
        if (is_ns_plain_safe(ch, c)) { ++pos; return true; }
        if (ch == '?' || ch == ':' || ch == '-') {
            if (pos + 1 < s.size() && is_ns_plain_safe(s[pos+1], c)) { ++pos; return true; }
        }
        return false;
    }
    bool parse_ns_plain_char(Context c) {
        char32_t ch = peek();
        if (ch == ':') {
            if (pos + 1 < s.size() && is_ns_plain_safe(s[pos+1], c)) { ++pos; return true; }
            return false;
        }
        if (ch == '#') {
            if (pos > 0 && is_ns_char(s[pos-1])) { ++pos; return true; }
            return false;
        }
        if (is_ns_plain_safe(ch, c)) { ++pos; return true; }
        return false;
    }
    bool parse_nb_ns_plain_in_line(Context c) {
        while (true) {
            size_t saved = pos;
            while (is_whitespace(peek())) ++pos;
            if (parse_ns_plain_char(c)) continue;
            pos = saved;
            break;
        }
        return true;
    }
    bool parse_ns_plain_one_line(Context c) {
        if (!parse_ns_plain_first(c)) return false;
        parse_nb_ns_plain_in_line(c);
        return true;
    }
    bool parse_s_ns_plain_next_line(int n, Context c) {
        if (!parse_s_flow_folded(n)) return false;
        if (!parse_ns_plain_char(c)) return false;
        parse_nb_ns_plain_in_line(c);
        return true;
    }
    bool parse_ns_plain_multi_line(int n, Context c) {
        if (!parse_ns_plain_one_line(c)) return false;
        while (true) {
            size_t saved = pos;
            if (parse_s_ns_plain_next_line(n, c)) continue;
            pos = saved;
            break;
        }
        return true;
    }
    bool parse_ns_plain(int n, Context c) {
        if (c == BLOCK_KEY || c == FLOW_KEY) return parse_ns_plain_one_line(c);
        return parse_ns_plain_multi_line(n, c);
    }

    Context in_flow_ctx(Context c) const {
        return (c == BLOCK_KEY || c == FLOW_KEY) ? FLOW_KEY : FLOW_IN;
    }
    bool parse_ns_flow_seq_entry(int n, Context c) {
        return parse_ns_flow_pair(n, c) || parse_ns_flow_node(n, c);
    }
    bool parse_ns_s_flow_seq_entries(int n, Context c) {
        if (!parse_ns_flow_seq_entry(n, c)) return false;
        parse_s_separate(n, c);
        if (eat(',')) {
            parse_s_separate(n, c);
            parse_ns_s_flow_seq_entries(n, c);
        }
        return true;
    }
    bool parse_c_flow_sequence(int n, Context c) {
        size_t saved = pos;
        if (!eat('[')) return false;
        parse_s_separate(n, c);
        Context inner = in_flow_ctx(c);
        if (!eat(']')) {
            if (parse_ns_s_flow_seq_entries(n, inner) && eat(']')) return true;
            pos = saved;
            return false;
        }
        return true;
    }
    bool parse_ns_flow_map_explicit_entry(int n, Context c) {
        if (parse_ns_flow_map_implicit_entry(n, c)) return true;
        return true; // e-node e-node
    }
    bool parse_ns_flow_map_implicit_entry(int n, Context c) {
        return parse_ns_flow_map_yaml_key_entry(n, c)
            || parse_c_ns_flow_map_empty_key_entry(n, c)
            || parse_c_ns_flow_map_json_key_entry(n, c);
    }
    bool parse_ns_flow_map_yaml_key_entry(int n, Context c) {
        size_t saved = pos;
        if (parse_ns_flow_yaml_node(n, c)) {
            size_t saved2 = pos;
            if (parse_s_separate(n, c) && parse_c_ns_flow_map_separate_value(n, c)) return true;
            pos = saved2;
            return true; // empty value
        }
        pos = saved;
        return false;
    }
    bool parse_c_ns_flow_map_empty_key_entry(int n, Context c) {
        return parse_c_ns_flow_map_separate_value(n, c);
    }
    bool parse_c_ns_flow_map_separate_value(int n, Context c) {
        size_t saved = pos;
        if (!eat(':')) return false;
        if (pos < s.size() && is_ns_plain_safe(s[pos], c)) { pos = saved; return false; }
        size_t saved2 = pos;
        if (parse_s_separate(n, c) && parse_ns_flow_node(n, c)) return true;
        pos = saved2;
        return true; // empty value
    }
    bool parse_c_ns_flow_map_json_key_entry(int n, Context c) {
        size_t saved = pos;
        if (parse_c_flow_json_node(n, c)) {
            size_t saved2 = pos;
            if (parse_s_separate(n, c) && parse_c_ns_flow_map_adjacent_value(n, c)) return true;
            pos = saved2;
            return true; // empty value
        }
        pos = saved;
        return false;
    }
    bool parse_c_ns_flow_map_adjacent_value(int n, Context c) {
        size_t saved = pos;
        if (!eat(':')) return false;
        size_t saved2 = pos;
        if (parse_s_separate(n, c) && parse_ns_flow_node(n, c)) return true;
        pos = saved2;
        return true; // empty value
    }
    bool parse_ns_flow_map_entry(int n, Context c) {
        size_t saved = pos;
        if (eat('?')) {
            if (parse_s_separate(n, c) && parse_ns_flow_map_explicit_entry(n, c)) return true;
        }
        pos = saved;
        return parse_ns_flow_map_implicit_entry(n, c);
    }
    bool parse_ns_s_flow_map_entries(int n, Context c) {
        if (!parse_ns_flow_map_entry(n, c)) return false;
        parse_s_separate(n, c);
        if (eat(',')) {
            parse_s_separate(n, c);
            parse_ns_s_flow_map_entries(n, c);
        }
        return true;
    }
    bool parse_c_flow_mapping(int n, Context c) {
        size_t saved = pos;
        if (!eat('{')) return false;
        parse_s_separate(n, c);
        Context inner = in_flow_ctx(c);
        if (!eat('}')) {
            if (parse_ns_s_flow_map_entries(n, inner) && eat('}')) return true;
            pos = saved;
            return false;
        }
        return true;
    }
    bool parse_ns_flow_pair_entry(int n, Context c) {
        return parse_ns_flow_pair_yaml_key_entry(n, c)
            || parse_c_ns_flow_map_empty_key_entry(n, c)
            || parse_c_ns_flow_pair_json_key_entry(n, c);
    }
    bool parse_ns_flow_pair(int n, Context c) {
        size_t saved = pos;
        if (eat('?')) {
            if (parse_s_separate(n, c) && parse_ns_flow_map_explicit_entry(n, c)) return true;
        }
        pos = saved;
        return parse_ns_flow_pair_entry(n, c);
    }
    bool parse_ns_flow_pair_yaml_key_entry(int n, Context c) {
        return parse_ns_s_implicit_yaml_key(c) && parse_c_ns_flow_map_separate_value(n, c);
    }
    bool parse_c_ns_flow_pair_json_key_entry(int n, Context c) {
        return parse_c_s_implicit_json_key(c) && parse_c_ns_flow_map_adjacent_value(n, c);
    }
    bool parse_ns_s_implicit_yaml_key(Context c) {
        size_t saved = pos;
        if (parse_ns_flow_yaml_node(0, c)) {
            if (!is_start_of_line() && is_whitespace(peek())) {
                while (is_whitespace(peek())) ++pos;
            }
            if (pos - saved > 1024) { pos = saved; return false; }
            return true;
        }
        pos = saved;
        return false;
    }
    bool parse_c_s_implicit_json_key(Context c) {
        size_t saved = pos;
        if (parse_c_flow_json_node(0, c)) {
            if (!is_start_of_line() && is_whitespace(peek())) {
                while (is_whitespace(peek())) ++pos;
            }
            if (pos - saved > 1024) { pos = saved; return false; }
            return true;
        }
        pos = saved;
        return false;
    }

    bool parse_ns_flow_yaml_content(int n, Context c) { return parse_ns_plain(n, c); }
    bool parse_c_flow_json_content(int n, Context c) {
        return parse_c_flow_sequence(n, c) || parse_c_flow_mapping(n, c)
            || parse_c_single_quoted(n, c) || parse_c_double_quoted(n, c);
    }
    bool parse_ns_flow_content(int n, Context c) {
        return parse_ns_flow_yaml_content(n, c) || parse_c_flow_json_content(n, c);
    }
    bool parse_ns_flow_yaml_node(int n, Context c) {
        if (parse_c_ns_alias_node()) return true;
        if (parse_ns_flow_yaml_content(n, c)) return true;
        size_t saved = pos;
        if (parse_c_ns_properties(n, c)) {
            size_t saved2 = pos;
            if (parse_s_separate(n, c) && parse_ns_flow_yaml_content(n, c)) return true;
            pos = saved2;
            return true; // e-scalar
        }
        pos = saved;
        return false;
    }
    bool parse_c_flow_json_node(int n, Context c) {
        size_t saved = pos;
        if (parse_c_ns_properties(n, c) && parse_s_separate(n, c)) {
            if (parse_c_flow_json_content(n, c)) return true;
            pos = saved;
            return false;
        }
        pos = saved;
        return parse_c_flow_json_content(n, c);
    }
    bool parse_ns_flow_node(int n, Context c) {
        if (parse_c_ns_alias_node()) return true;
        if (parse_ns_flow_content(n, c)) return true;
        size_t saved = pos;
        if (parse_c_ns_properties(n, c)) {
            size_t saved2 = pos;
            if (parse_s_separate(n, c) && parse_ns_flow_content(n, c)) return true;
            pos = saved2;
            return true; // e-scalar
        }
        pos = saved;
        return false;
    }

    bool parse_c_indentation_indicator(int& m) {
        char32_t c = peek();
        if (c >= '1' && c <= '9') { m = c - '0'; ++pos; return true; }
        return false;
    }
    bool parse_c_chomping_indicator(Chomp& t) {
        if (eat('-')) { t = STRIP; return true; }
        if (eat('+')) { t = KEEP; return true; }
        return false;
    }
    bool parse_c_b_block_header(Chomp& t, int& m) {
        m = -1; t = CLIP;
        size_t saved = pos;
        if (parse_c_indentation_indicator(m)) {
            if (parse_c_chomping_indicator(t)) {
                if (parse_s_b_comment()) return true;
            } else {
                if (parse_s_b_comment()) { t = CLIP; return true; }
            }
        }
        pos = saved;
        if (parse_c_chomping_indicator(t)) {
            int m2;
            if (parse_c_indentation_indicator(m2)) {
                m = m2;
                if (parse_s_b_comment()) return true;
            } else {
                if (parse_s_b_comment()) { m = -1; return true; }
            }
        }
        pos = saved;
        if (parse_s_b_comment()) { t = CLIP; m = -1; return true; }
        return false;
    }
    int detect_block_scalar_indent(int n) {
        size_t saved = pos;
        int max_spaces = 0;
        while (true) {
            int spaces = 0;
            while (eat(' ')) ++spaces;
            if (eof()) { pos = saved; return max_spaces; }
            char32_t c = peek();
            if (c == '\n' || c == '\r') {
                if (spaces > max_spaces) max_spaces = spaces;
                match_break();
                continue;
            }
            if (spaces <= n) { pos = saved; return max_spaces; }
            pos = saved;
            return spaces;
        }
    }
    bool parse_l_nb_literal_text(int n) {
        while (parse_l_empty(n, BLOCK_IN)) {}
        if (!match_indent(n)) return false;
        if (eof() || !is_nb_char(peek())) return false;
        while (!eof() && is_nb_char(peek())) ++pos;
        return true;
    }
    bool parse_b_nb_literal_next(int n) {
        return match_break() && parse_l_nb_literal_text(n);
    }
    bool match_b_chomped_last(Chomp) {
        return match_break() || eof();
    }
    bool parse_l_trail_comments(int n) {
        size_t saved = pos;
        if (!match_indent_less_than(n)) { pos = saved; return false; }
        if (!parse_c_nb_comment_text()) { pos = saved; return false; }
        if (!parse_b_comment()) { pos = saved; return false; }
        while (parse_l_comment()) {}
        return true;
    }
    bool parse_l_chomped_empty(int n, Chomp) {
        while (true) {
            size_t saved = pos;
            if (match_indent_less_or_equal(n) && match_break()) continue;
            pos = saved;
            break;
        }
        parse_l_trail_comments(n);
        return true;
    }
    bool parse_l_literal_content(int n, Chomp t) {
        if (parse_l_nb_literal_text(n)) {
            while (true) {
                size_t saved = pos;
                if (match_break() && parse_l_nb_literal_text(n)) continue;
                pos = saved;
                break;
            }
            if (!match_b_chomped_last(t)) return false;
        }
        return parse_l_chomped_empty(n, t);
    }
    bool parse_c_l_literal(int n) {
        if (!eat('|')) return false;
        Chomp t; int m;
        if (!parse_c_b_block_header(t, m)) return false;
        int cn = (m >= 0) ? n + m : detect_block_scalar_indent(n);
        if (cn < 0) cn = 0;
        return parse_l_literal_content(cn, t);
    }
    bool parse_s_nb_folded_text(int n) {
        if (!match_indent(n)) return false;
        if (eof() || !is_ns_char(peek())) return false;
        ++pos;
        while (!eof() && is_nb_char(peek())) ++pos;
        return true;
    }
    bool parse_b_l_folded(int n, Context c) {
        size_t saved = pos;
        if (match_break()) {
            if (parse_l_empty(n, c)) {
                while (parse_l_empty(n, c)) {}
                return true;
            }
            return true; // b-as-space
        }
        return false;
    }
    bool parse_l_nb_folded_lines(int n) {
        if (!parse_s_nb_folded_text(n)) return false;
        while (true) {
            size_t saved = pos;
            if (parse_b_l_folded(n, BLOCK_IN) && parse_s_nb_folded_text(n)) continue;
            pos = saved;
            break;
        }
        return true;
    }
    bool parse_s_nb_spaced_text(int n) {
        if (!match_indent(n)) return false;
        if (eof() || !is_whitespace(peek())) return false;
        ++pos;
        while (!eof() && is_nb_char(peek())) ++pos;
        return true;
    }
    bool parse_b_l_spaced(int n) {
        if (!match_break()) return false;
        while (parse_l_empty(n, BLOCK_IN)) {}
        return true;
    }
    bool parse_l_nb_spaced_lines(int n) {
        if (!parse_s_nb_spaced_text(n)) return false;
        while (true) {
            size_t saved = pos;
            if (parse_b_l_spaced(n) && parse_s_nb_spaced_text(n)) continue;
            pos = saved;
            break;
        }
        return true;
    }
    bool parse_l_nb_same_lines(int n) {
        while (parse_l_empty(n, BLOCK_IN)) {}
        return parse_l_nb_folded_lines(n) || parse_l_nb_spaced_lines(n);
    }
    bool parse_l_nb_diff_lines(int n) {
        if (!parse_l_nb_same_lines(n)) return false;
        while (true) {
            size_t saved = pos;
            if (match_break() && parse_l_nb_same_lines(n)) continue;
            pos = saved;
            break;
        }
        return true;
    }
    bool parse_l_folded_content(int n, Chomp t) {
        if (parse_l_nb_diff_lines(n)) {
            if (!match_b_chomped_last(t)) return false;
        }
        return parse_l_chomped_empty(n, t);
    }
    bool parse_c_l_folded(int n) {
        if (!eat('>')) return false;
        Chomp t; int m;
        if (!parse_c_b_block_header(t, m)) return false;
        int cn = (m >= 0) ? n + m : detect_block_scalar_indent(n);
        if (cn < 0) cn = 0;
        return parse_l_folded_content(cn, t);
    }

    bool parse_l_empty(int n, Context c) {
        size_t saved = pos;
        if (match_s_line_prefix(n, c)) {
            if (c == FLOW_OUT || c == FLOW_IN || c == FLOW_KEY) {
                while (eat(' ')) {}
            }
            if (match_break()) return true;
        }
        pos = saved;
        if (match_indent_less_than(n)) {
            if (match_break()) return true;
        }
        pos = saved;
        return false;
    }

    bool parse_c_l_block_seq_entry(int n) {
        size_t saved = pos;
        if (!eat('-')) return false;
        if (pos < s.size() && is_ns_char(peek())) { pos = saved; return false; }
        return parse_s_l_block_indented(n, BLOCK_IN);
    }
    bool parse_ns_l_compact_sequence(int n) {
        size_t saved = pos;
        if (!parse_c_l_block_seq_entry(n)) { pos = saved; return false; }
        while (true) {
            size_t saved2 = pos;
            if (match_indent(n) && parse_c_l_block_seq_entry(n)) continue;
            pos = saved2;
            break;
        }
        return true;
    }
    bool parse_ns_l_compact_mapping(int n) {
        size_t saved = pos;
        if (!parse_ns_l_block_map_entry(n)) { pos = saved; return false; }
        while (true) {
            size_t saved2 = pos;
            if (match_indent(n) && parse_ns_l_block_map_entry(n)) continue;
            pos = saved2;
            break;
        }
        return true;
    }
    bool parse_s_l_block_indented(int n, Context c) {
        size_t pos0 = pos;
        size_t pos_compact = pos;
        int m = 0;
        while (eat(' ')) {
            ++m;
            size_t saved = pos;
            if (parse_ns_l_compact_sequence(n + 1 + m) || parse_ns_l_compact_mapping(n + 1 + m)) {
                return true;
            }
            pos = saved;
        }
        pos = pos0;
        if (parse_s_l_block_node(n, c)) return true;
        pos = pos0;
        if (parse_s_l_comments()) return true;
        pos = pos0;
        return false;
    }
    bool parse_l_block_sequence(int n) {
        size_t saved = pos;
        for (int m = 0; m < 200; ++m) {
            pos = saved;
            int indent = n + 1 + m;
            if (indent < 0) indent = 0;
            if (match_indent(indent) && parse_c_l_block_seq_entry(indent)) {
                while (true) {
                    size_t saved2 = pos;
                    if (match_indent(indent) && parse_c_l_block_seq_entry(indent)) continue;
                    pos = saved2;
                    break;
                }
                return true;
            }
        }
        pos = saved;
        return false;
    }
    bool parse_c_l_block_map_explicit_key(int n) {
        size_t saved = pos;
        if (!eat('?')) return false;
        if (pos < s.size() && is_ns_char(peek())) { pos = saved; return false; }
        if (parse_s_l_block_indented(n, BLOCK_OUT)) return true;
        pos = saved;
        return false;
    }
    bool parse_l_block_map_explicit_value(int n) {
        size_t saved = pos;
        if (!match_indent(n)) return false;
        if (!eat(':')) { pos = saved; return false; }
        if (pos < s.size() && !is_whitespace(peek()) && peek() != '\n' && peek() != '\r') {
            pos = saved; return false;
        }
        if (parse_s_l_block_indented(n, BLOCK_OUT)) return true;
        pos = saved;
        return false;
    }
    bool parse_c_l_block_map_explicit_entry(int n) {
        if (!parse_c_l_block_map_explicit_key(n)) return false;
        if (parse_l_block_map_explicit_value(n)) return true;
        return true; // empty value
    }
    bool parse_ns_s_block_map_implicit_key() {
        return parse_c_s_implicit_json_key(BLOCK_KEY) || parse_ns_s_implicit_yaml_key(BLOCK_KEY);
    }
    bool parse_c_l_block_map_implicit_value(int n) {
        size_t saved = pos;
        if (!eat(':')) return false;
        if (pos < s.size() && !is_whitespace(peek()) && peek() != '\n' && peek() != '\r') {
            pos = saved; return false;
        }
        size_t saved2 = pos;
        if (parse_s_l_block_node(n, BLOCK_OUT)) return true;
        pos = saved2;
        if (parse_s_l_comments()) return true;
        pos = saved;
        return false;
    }
    bool parse_ns_l_block_map_implicit_entry(int n) {
        size_t saved = pos;
        bool has_key = parse_ns_s_block_map_implicit_key();
        if (!has_key) {
            // e-node empty key
        }
        if (parse_c_l_block_map_implicit_value(n)) return true;
        pos = saved;
        return false;
    }
    bool parse_ns_l_block_map_entry(int n) {
        return parse_c_l_block_map_explicit_entry(n) || parse_ns_l_block_map_implicit_entry(n);
    }
    bool parse_l_block_mapping(int n) {
        size_t saved = pos;
        for (int m = 0; m < 200; ++m) {
            pos = saved;
            int indent = n + 1 + m;
            if (indent < 0) indent = 0;
            if (match_indent(indent) && parse_ns_l_block_map_entry(indent)) {
                while (true) {
                    size_t saved2 = pos;
                    if (match_indent(indent) && parse_ns_l_block_map_entry(indent)) continue;
                    pos = saved2;
                    break;
                }
                return true;
            }
        }
        pos = saved;
        return false;
    }

    bool parse_s_l_flow_in_block(int n) {
        size_t saved = pos;
        if (parse_s_separate(n + 1, FLOW_OUT)) {
            if (parse_ns_flow_node(n + 1, FLOW_OUT)) {
                if (parse_s_l_comments()) return true;
            }
        }
        pos = saved;
        return false;
    }
    bool parse_s_l_block_scalar(int n, Context c) {
        size_t saved = pos;
        if (parse_s_separate(n + 1, c)) {
            size_t saved2 = pos;
            if (parse_c_ns_properties(n + 1, c)) {
                if (parse_s_separate(n + 1, c)) {
                    if (parse_c_l_literal(n) || parse_c_l_folded(n)) return true;
                }
            }
            pos = saved2;
            if (parse_c_l_literal(n) || parse_c_l_folded(n)) return true;
        }
        pos = saved;
        return false;
    }
    bool parse_seq_space(int n, Context c) {
        if (c == BLOCK_OUT) return parse_l_block_sequence(n - 1);
        if (c == BLOCK_IN) return parse_l_block_sequence(n);
        return false;
    }
    bool parse_s_l_block_collection(int n, Context c) {
        size_t saved = pos;
        if (parse_s_separate(n + 1, c) && parse_c_ns_properties(n + 1, c)) {
            if (parse_s_l_comments()) {
                if (parse_seq_space(n, c) || parse_l_block_mapping(n)) return true;
            }
            pos = saved;
            return false;
        }
        if (parse_s_l_comments()) {
            if (parse_seq_space(n, c) || parse_l_block_mapping(n)) return true;
        }
        pos = saved;
        return false;
    }
    bool parse_s_l_block_in_block(int n, Context c) {
        return parse_s_l_block_scalar(n, c) || parse_s_l_block_collection(n, c);
    }
    bool parse_s_l_block_node(int n, Context c) {
        return parse_s_l_block_in_block(n, c) || parse_s_l_flow_in_block(n);
    }

    bool parse_l_document_prefix() {
        eat(0xFEFF);
        while (parse_l_comment()) {}
        return true;
    }
    bool parse_c_directives_end() { return match_str("---"); }
    bool parse_c_document_end() {
        size_t saved = pos;
        if (match_str("...")) {
            if (eof()) return true;
            char32_t c = peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return true;
        }
        pos = saved;
        return false;
    }
    bool parse_l_document_suffix() {
        size_t saved = pos;
        if (!parse_c_document_end()) return false;
        if (!parse_s_l_comments()) { pos = saved; return false; }
        return true;
    }
    bool check_forbidden(size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            bool bol = (i == 0);
            if (!bol) {
                char32_t pr = s[i-1];
                if (pr == '\n' || pr == '\r') bol = true;
            }
            if (!bol) continue;
            if (i + 2 < end && s[i] == '-' && s[i+1] == '-' && s[i+2] == '-') {
                size_t a = i + 3;
                if (a >= end) return false;
                char32_t nxt = s[a];
                if (nxt == '\n' || nxt == '\r' || nxt == ' ' || nxt == '\t') return false;
            }
            if (i + 2 < end && s[i] == '.' && s[i+1] == '.' && s[i+2] == '.') {
                size_t a = i + 3;
                if (a >= end) return false;
                char32_t nxt = s[a];
                if (nxt == '\n' || nxt == '\r' || nxt == ' ' || nxt == '\t') return false;
            }
        }
        return true;
    }
    bool parse_l_bare_document() {
        size_t start = pos;
        size_t saved = pos;
        if (parse_s_l_block_node(-1, BLOCK_IN)) {
            if (check_forbidden(start, pos)) return true;
        }
        pos = saved;
        return false;
    }
    bool parse_l_explicit_document() {
        size_t saved = pos;
        if (!parse_c_directives_end()) return false;
        if (parse_l_bare_document()) return true;
        if (parse_s_l_comments()) return true;
        pos = saved;
        return false;
    }
    bool parse_l_directive_document() {
        size_t saved = pos;
        if (!parse_l_directive()) { pos = saved; return false; }
        while (parse_l_directive()) {}
        if (parse_l_explicit_document()) return true;
        pos = saved;
        return false;
    }
    bool parse_l_any_document() {
        return parse_l_directive_document() || parse_l_explicit_document() || parse_l_bare_document();
    }
    bool parse_l_yaml_stream() {
        while (parse_l_document_prefix()) {}
        parse_l_any_document();
        while (true) {
            size_t saved = pos;
            if (parse_l_document_suffix()) {
                while (parse_l_document_suffix()) {}
                while (parse_l_document_prefix()) {}
                parse_l_any_document();
                continue;
            }
            pos = saved;
            if (eat(0xFEFF)) { continue; }
            pos = saved;
            if (parse_l_comment()) { continue; }
            pos = saved;
            if (parse_l_explicit_document()) { continue; }
            pos = saved;
            break;
        }
        return true;
    }

    bool parse() {
        if (!parse_l_yaml_stream()) return false;
        while (!eof() && is_whitespace(peek())) ++pos;
        return eof();
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<uint8_t> bytes;
    char ch;
    while (cin.get(ch)) bytes.push_back(static_cast<uint8_t>(ch));
    vector<char32_t> chars = decode(bytes);
    Parser p(chars);
    bool ok = p.parse();
    cout << (ok ? "valid" : "invalid") << "\n";
    return 0;
}
