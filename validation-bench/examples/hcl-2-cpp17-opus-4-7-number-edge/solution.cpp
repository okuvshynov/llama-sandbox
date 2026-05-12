#include <iostream>
#include <string>
#include <sstream>
#include <cctype>

class Parser {
public:
    std::string src;
    size_t pos = 0;
    bool ok = true;
    bool nlIsWS = false;

    explicit Parser(std::string s) : src(std::move(s)) {}

    bool atEnd() const { return pos >= src.size(); }
    char peek(size_t o = 0) const { return pos + o < src.size() ? src[pos + o] : '\0'; }

    static bool isIdStart(unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c >= 0x80;
    }
    static bool isIdCont(unsigned char c) {
        return isIdStart(c) || (c >= '0' && c <= '9') || c == '-';
    }
    static bool isHex(unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }
    static bool isDig(unsigned char c) { return c >= '0' && c <= '9'; }

    bool skipWS() {
        bool crossed = false;
        while (!atEnd()) {
            char c = peek();
            if (c == ' ' || c == '\t') { pos++; continue; }
            if (c == '/' && peek(1) == '*') {
                pos += 2;
                bool found = false;
                while (pos + 1 < src.size()) {
                    if (src[pos] == '*' && src[pos + 1] == '/') { pos += 2; found = true; break; }
                    if (src[pos] == '\n') crossed = true;
                    pos++;
                }
                if (!found) { ok = false; return crossed; }
                continue;
            }
            if (nlIsWS) {
                if (c == '\n') { pos++; crossed = true; continue; }
                if (c == '\r' && peek(1) == '\n') { pos += 2; crossed = true; continue; }
                if (c == '#' || (c == '/' && peek(1) == '/')) {
                    while (!atEnd() && peek() != '\n') pos++;
                    crossed = true;
                    continue;
                }
            }
            break;
        }
        return crossed;
    }

    bool parseIdent(std::string& out) {
        if (atEnd() || !isIdStart((unsigned char)peek())) return false;
        size_t start = pos;
        pos++;
        while (!atEnd() && isIdCont((unsigned char)peek())) pos++;
        out.assign(src, start, pos - start);
        return true;
    }

    bool peekIdent(std::string& out, size_t& endP) const {
        size_t p = pos;
        if (p >= src.size() || !isIdStart((unsigned char)src[p])) return false;
        p++;
        while (p < src.size() && isIdCont((unsigned char)src[p])) p++;
        out.assign(src, pos, p - pos);
        endP = p;
        return true;
    }

    bool parseNumber() {
        if (atEnd() || !isDig((unsigned char)peek())) return false;
        while (!atEnd() && isDig((unsigned char)peek())) pos++;
        if (peek() == '.' && isDig((unsigned char)peek(1))) {
            pos++;
            while (!atEnd() && isDig((unsigned char)peek())) pos++;
        }
        if (peek() == 'e' || peek() == 'E') {
            size_t save = pos;
            pos++;
            if (peek() == '+' || peek() == '-') pos++;
            if (!isDig((unsigned char)peek())) {
                pos = save;
            } else {
                while (!atEnd() && isDig((unsigned char)peek())) pos++;
            }
        }
        return true;
    }

    bool isHeredocEnd(const std::string& marker, bool indented, size_t& endP) const {
        size_t p = pos;
        if (indented) {
            while (p < src.size() && (src[p] == ' ' || src[p] == '\t')) p++;
        }
        if (p + marker.size() > src.size()) return false;
        if (src.compare(p, marker.size(), marker) != 0) return false;
        p += marker.size();
        if (p < src.size() && isIdCont((unsigned char)src[p])) return false;
        size_t pSave = p;
        while (p < src.size() && (src[p] == ' ' || src[p] == '\t')) p++;
        if (p == src.size()) { endP = pSave; return true; }
        if (src[p] == '\n' || (src[p] == '\r' && p + 1 < src.size() && src[p + 1] == '\n')) {
            endP = pSave; return true;
        }
        return false;
    }

    bool parseExpression() {
        skipWS();
        return parseConditional();
    }

    bool parseConditional() {
        if (!parseBinaryOp(1)) return false;
        size_t savePos = pos;
        skipWS();
        if (peek() == '?') {
            pos++;
            skipWS();
            if (!parseExpression()) return false;
            skipWS();
            if (peek() != ':') return false;
            pos++;
            skipWS();
            if (!parseExpression()) return false;
        } else {
            pos = savePos;
        }
        return true;
    }

    int binopPrec(int& opLen) {
        opLen = 0;
        if (atEnd()) return 0;
        char c = peek(), d = peek(1);
        if (c == '|' && d == '|') { opLen = 2; return 1; }
        if (c == '&' && d == '&') { opLen = 2; return 2; }
        if (c == '=' && d == '=') { opLen = 2; return 3; }
        if (c == '!' && d == '=') { opLen = 2; return 3; }
        if (c == '<' && d == '=') { opLen = 2; return 4; }
        if (c == '>' && d == '=') { opLen = 2; return 4; }
        if (c == '<') { opLen = 1; return 4; }
        if (c == '>') { opLen = 1; return 4; }
        if (c == '+') { opLen = 1; return 5; }
        if (c == '-') { opLen = 1; return 5; }
        if (c == '*') { opLen = 1; return 6; }
        if (c == '/') {
            if (d == '*' || d == '/') return 0;
            opLen = 1; return 6;
        }
        if (c == '%') {
            if (d == '{') return 0;
            opLen = 1; return 6;
        }
        return 0;
    }

    bool parseBinaryOp(int minPrec) {
        if (!parseUnary()) return false;
        while (true) {
            size_t savePos = pos;
            skipWS();
            int opLen = 0;
            int prec = binopPrec(opLen);
            if (prec < minPrec) {
                pos = savePos;
                break;
            }
            pos += opLen;
            skipWS();
            if (!parseBinaryOp(prec + 1)) return false;
        }
        return true;
    }

    bool parseUnary() {
        skipWS();
        if (peek() == '-' || peek() == '!') {
            pos++;
            return parseUnary();
        }
        return parsePostfix();
    }

    bool parsePostfix() {
        if (!parsePrimary()) return false;
        while (true) {
            char c = peek();
            if (c == '.') {
                size_t savePos = pos;
                pos++;
                if (isDig((unsigned char)peek())) {
                    while (!atEnd() && isDig((unsigned char)peek())) pos++;
                } else if (peek() == '*') {
                    pos++;
                    while (peek() == '.') {
                        size_t save2 = pos;
                        pos++;
                        std::string ign;
                        if (!parseIdent(ign)) { pos = save2; break; }
                    }
                } else {
                    std::string ign;
                    if (!parseIdent(ign)) {
                        pos = savePos;
                        break;
                    }
                }
            } else if (c == '[') {
                pos++;
                bool save = nlIsWS; nlIsWS = true;
                skipWS();
                if (peek() == '*') {
                    pos++;
                    skipWS();
                    if (peek() != ']') return false;
                    pos++;
                    nlIsWS = save;
                    while (true) {
                        if (peek() == '.') {
                            size_t save2 = pos;
                            pos++;
                            if (isDig((unsigned char)peek())) {
                                while (!atEnd() && isDig((unsigned char)peek())) pos++;
                            } else {
                                std::string ign;
                                if (!parseIdent(ign)) { pos = save2; break; }
                            }
                        } else if (peek() == '[') {
                            pos++;
                            bool s2 = nlIsWS; nlIsWS = true;
                            skipWS();
                            if (!parseExpression()) return false;
                            skipWS();
                            if (peek() != ']') return false;
                            pos++;
                            nlIsWS = s2;
                        } else break;
                    }
                } else {
                    if (!parseExpression()) return false;
                    skipWS();
                    if (peek() != ']') return false;
                    pos++;
                    nlIsWS = save;
                }
            } else break;
        }
        return true;
    }

    bool parsePrimary() {
        skipWS();
        if (atEnd()) return false;
        char c = peek();

        if (isDig((unsigned char)c)) return parseNumber();

        if (c == '"') {
            pos++;
            return parseTmpl(true, "", false, false, false, false) == 0;
        }
        if (c == '<' && peek(1) == '<') return parseHeredoc();

        if (c == '(') {
            pos++;
            bool save = nlIsWS; nlIsWS = true;
            skipWS();
            if (!parseExpression()) return false;
            skipWS();
            if (peek() != ')') return false;
            pos++;
            nlIsWS = save;
            return true;
        }

        if (c == '[') return parseTupleOrForTuple();
        if (c == '{') return parseObjectOrForObject();

        if (isIdStart((unsigned char)c)) {
            std::string id;
            if (!parseIdent(id)) return false;
            if (peek() == '(') {
                pos++;
                bool save = nlIsWS; nlIsWS = true;
                skipWS();
                if (peek() == ')') { pos++; nlIsWS = save; return true; }
                while (true) {
                    if (!parseExpression()) return false;
                    skipWS();
                    if (peek() == ',') {
                        pos++;
                        skipWS();
                        if (peek() == ')') { pos++; nlIsWS = save; return true; }
                        continue;
                    }
                    if (peek(0) == '.' && peek(1) == '.' && peek(2) == '.') {
                        pos += 3;
                        skipWS();
                        if (peek() != ')') return false;
                        pos++;
                        nlIsWS = save;
                        return true;
                    }
                    if (peek() == ')') {
                        pos++;
                        nlIsWS = save;
                        return true;
                    }
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    bool parseTupleOrForTuple() {
        pos++;
        bool save = nlIsWS; nlIsWS = true;
        skipWS();
        if (peek() == ']') { pos++; nlIsWS = save; return true; }

        if (isIdStart((unsigned char)peek())) {
            std::string id;
            size_t ep;
            if (peekIdent(id, ep) && id == "for") {
                pos = ep;
                skipWS();
                std::string id1;
                if (!parseIdent(id1)) return false;
                skipWS();
                if (peek() == ',') {
                    pos++; skipWS();
                    std::string id2;
                    if (!parseIdent(id2)) return false;
                    skipWS();
                }
                std::string inkw;
                if (!parseIdent(inkw) || inkw != "in") return false;
                skipWS();
                if (!parseExpression()) return false;
                skipWS();
                if (peek() != ':') return false;
                pos++;
                skipWS();
                if (!parseExpression()) return false;
                skipWS();
                if (isIdStart((unsigned char)peek())) {
                    std::string iw;
                    size_t ep2;
                    if (peekIdent(iw, ep2) && iw == "if") {
                        pos = ep2;
                        skipWS();
                        if (!parseExpression()) return false;
                        skipWS();
                    }
                }
                if (peek() != ']') return false;
                pos++;
                nlIsWS = save;
                return true;
            }
        }

        if (!parseExpression()) return false;
        while (true) {
            bool crossed = skipWS();
            if (peek() == ']') { pos++; nlIsWS = save; return true; }
            if (peek() == ',') {
                pos++;
                skipWS();
                if (peek() == ']') { pos++; nlIsWS = save; return true; }
                if (!parseExpression()) return false;
                continue;
            }
            if (crossed) {
                if (!parseExpression()) return false;
                continue;
            }
            return false;
        }
    }

    bool parseObjectOrForObject() {
        pos++;
        bool save = nlIsWS; nlIsWS = true;
        skipWS();
        if (peek() == '}') { pos++; nlIsWS = save; return true; }

        if (isIdStart((unsigned char)peek())) {
            std::string id;
            size_t ep;
            if (peekIdent(id, ep) && id == "for") {
                pos = ep;
                skipWS();
                std::string id1;
                if (!parseIdent(id1)) return false;
                skipWS();
                if (peek() == ',') {
                    pos++; skipWS();
                    std::string id2;
                    if (!parseIdent(id2)) return false;
                    skipWS();
                }
                std::string inkw;
                if (!parseIdent(inkw) || inkw != "in") return false;
                skipWS();
                if (!parseExpression()) return false;
                skipWS();
                if (peek() != ':') return false;
                pos++;
                skipWS();
                if (!parseExpression()) return false;
                skipWS();
                if (peek() != '=' || peek(1) != '>') return false;
                pos += 2;
                skipWS();
                if (!parseExpression()) return false;
                skipWS();
                if (peek(0) == '.' && peek(1) == '.' && peek(2) == '.') { pos += 3; skipWS(); }
                if (isIdStart((unsigned char)peek())) {
                    std::string iw;
                    size_t ep2;
                    if (peekIdent(iw, ep2) && iw == "if") {
                        pos = ep2;
                        skipWS();
                        if (!parseExpression()) return false;
                        skipWS();
                    }
                }
                if (peek() != '}') return false;
                pos++;
                nlIsWS = save;
                return true;
            }
        }

        if (!parseObjectElem()) return false;
        while (true) {
            bool crossed = skipWS();
            if (peek() == '}') { pos++; nlIsWS = save; return true; }
            if (peek() == ',') {
                pos++;
                skipWS();
                if (peek() == '}') { pos++; nlIsWS = save; return true; }
                if (!parseObjectElem()) return false;
                continue;
            }
            if (crossed) {
                if (!parseObjectElem()) return false;
                continue;
            }
            return false;
        }
    }

    bool parseObjectElem() {
        if (!parseExpression()) return false;
        skipWS();
        if (peek() == '=' && peek(1) == '=') return false;
        if (peek() == '=' && peek(1) == '>') return false;
        if (peek() != '=' && peek() != ':') return false;
        pos++;
        skipWS();
        if (!parseExpression()) return false;
        return true;
    }

    bool parseHeredoc() {
        pos += 2;
        bool indented = false;
        if (peek() == '-') { pos++; indented = true; }
        std::string marker;
        if (!parseIdent(marker)) return false;
        while (peek() == ' ' || peek() == '\t') pos++;
        if (peek() == '\n') pos++;
        else if (peek() == '\r' && peek(1) == '\n') pos += 2;
        else return false;
        int r = parseTmpl(false, marker, indented, false, false, false);
        return r == 0;
    }

    int parseTmpl(bool quoted, const std::string& hMark, bool hIndent,
                  bool aElse, bool aEndIf, bool aEndFor) {
        bool atLineStart = !quoted;
        while (!atEnd()) {
            if (!quoted && atLineStart) {
                size_t endP;
                if (isHeredocEnd(hMark, hIndent, endP)) {
                    pos = endP;
                    return 0;
                }
            }
            char c = peek();
            if (quoted) {
                if (c == '"') { pos++; return 0; }
                if (c == '\n' || c == '\r') return -1;
                if (c == '\\') {
                    pos++;
                    if (atEnd()) return -1;
                    char e = peek();
                    if (e == 'n' || e == 'r' || e == 't' || e == '"' || e == '\\') {
                        pos++;
                    } else if (e == 'u') {
                        pos++;
                        for (int i = 0; i < 4; i++) {
                            if (atEnd() || !isHex((unsigned char)peek())) return -1;
                            pos++;
                        }
                    } else if (e == 'U') {
                        pos++;
                        for (int i = 0; i < 8; i++) {
                            if (atEnd() || !isHex((unsigned char)peek())) return -1;
                            pos++;
                        }
                    } else {
                        return -1;
                    }
                    continue;
                }
            }

            if (c == '$' && peek(1) == '$' && peek(2) == '{') {
                pos += 3;
                atLineStart = false;
                continue;
            }
            if (c == '%' && peek(1) == '%' && peek(2) == '{') {
                pos += 3;
                atLineStart = false;
                continue;
            }
            if (c == '$' && peek(1) == '{') {
                pos += 2;
                if (peek() == '~') pos++;
                bool savNL = nlIsWS; nlIsWS = true;
                skipWS();
                if (!parseExpression()) { nlIsWS = savNL; return -1; }
                skipWS();
                nlIsWS = savNL;
                if (peek() == '~') pos++;
                if (peek() != '}') return -1;
                pos++;
                atLineStart = false;
                continue;
            }
            if (c == '%' && peek(1) == '{') {
                pos += 2;
                if (peek() == '~') pos++;
                bool savNL = nlIsWS; nlIsWS = true;
                skipWS();
                std::string kw;
                if (!parseIdent(kw)) { nlIsWS = savNL; return -1; }
                if (kw == "if") {
                    skipWS();
                    if (!parseExpression()) { nlIsWS = savNL; return -1; }
                    skipWS();
                    nlIsWS = savNL;
                    if (peek() == '~') pos++;
                    if (peek() != '}') return -1;
                    pos++;
                    int r = parseTmpl(quoted, hMark, hIndent, true, true, false);
                    if (r == 1) {
                        int r2 = parseTmpl(quoted, hMark, hIndent, false, true, false);
                        if (r2 != 2) return -1;
                    } else if (r != 2) {
                        return -1;
                    }
                    atLineStart = false;
                    continue;
                } else if (kw == "else") {
                    if (!aElse) { nlIsWS = savNL; return -1; }
                    skipWS();
                    nlIsWS = savNL;
                    if (peek() == '~') pos++;
                    if (peek() != '}') return -1;
                    pos++;
                    return 1;
                } else if (kw == "endif") {
                    if (!aEndIf) { nlIsWS = savNL; return -1; }
                    skipWS();
                    nlIsWS = savNL;
                    if (peek() == '~') pos++;
                    if (peek() != '}') return -1;
                    pos++;
                    return 2;
                } else if (kw == "for") {
                    skipWS();
                    std::string id1;
                    if (!parseIdent(id1)) { nlIsWS = savNL; return -1; }
                    skipWS();
                    if (peek() == ',') {
                        pos++; skipWS();
                        std::string id2;
                        if (!parseIdent(id2)) { nlIsWS = savNL; return -1; }
                        skipWS();
                    }
                    std::string inkw;
                    if (!parseIdent(inkw) || inkw != "in") { nlIsWS = savNL; return -1; }
                    skipWS();
                    if (!parseExpression()) { nlIsWS = savNL; return -1; }
                    skipWS();
                    nlIsWS = savNL;
                    if (peek() == '~') pos++;
                    if (peek() != '}') return -1;
                    pos++;
                    int r = parseTmpl(quoted, hMark, hIndent, false, false, true);
                    if (r != 3) return -1;
                    atLineStart = false;
                    continue;
                } else if (kw == "endfor") {
                    if (!aEndFor) { nlIsWS = savNL; return -1; }
                    skipWS();
                    nlIsWS = savNL;
                    if (peek() == '~') pos++;
                    if (peek() != '}') return -1;
                    pos++;
                    return 3;
                } else {
                    nlIsWS = savNL;
                    return -1;
                }
            }
            if (c == '\n') {
                pos++;
                atLineStart = true;
                continue;
            }
            if (c == '\r') {
                pos++;
                if (peek() == '\n') pos++;
                atLineStart = true;
                continue;
            }
            pos++;
            atLineStart = false;
        }
        return -1;
    }

    void consumeBlankLines() {
        while (!atEnd()) {
            while (!atEnd()) {
                char c = peek();
                if (c == ' ' || c == '\t') pos++;
                else if (c == '/' && peek(1) == '*') {
                    pos += 2;
                    bool f = false;
                    while (pos + 1 < src.size()) {
                        if (src[pos] == '*' && src[pos + 1] == '/') { pos += 2; f = true; break; }
                        pos++;
                    }
                    if (!f) { ok = false; return; }
                }
                else break;
            }
            char c = peek();
            if (c == '\n') pos++;
            else if (c == '\r' && peek(1) == '\n') pos += 2;
            else if (c == '\r') pos++;
            else if (c == '#' || (c == '/' && peek(1) == '/')) {
                while (!atEnd() && peek() != '\n') pos++;
            }
            else break;
        }
    }

    bool parseBody() {
        bool save = nlIsWS;
        nlIsWS = false;
        while (true) {
            consumeBlankLines();
            if (atEnd() || peek() == '}') { nlIsWS = save; return true; }
            if (!parseAttributeOrBlock()) { nlIsWS = save; return false; }
            skipWS();
            if (atEnd() || peek() == '}') { nlIsWS = save; return true; }
            char c = peek();
            if (c == '\n') pos++;
            else if (c == '\r' && peek(1) == '\n') pos += 2;
            else if (c == '\r') pos++;
            else if (c == '#' || (c == '/' && peek(1) == '/')) {
                while (!atEnd() && peek() != '\n') pos++;
            }
            else { nlIsWS = save; return false; }
        }
    }

    bool parseAttributeOrBlock() {
        std::string id;
        if (!parseIdent(id)) return false;
        skipWS();
        if (peek() == '=' && peek(1) != '=') {
            pos++;
            skipWS();
            if (!parseExpression()) return false;
            return true;
        }
        while (peek() != '{') {
            char c = peek();
            if (c == '"') {
                pos++;
                if (parseTmpl(true, "", false, false, false, false) != 0) return false;
            } else if (isIdStart((unsigned char)c)) {
                std::string lbl;
                if (!parseIdent(lbl)) return false;
            } else return false;
            skipWS();
        }
        pos++;
        skipWS();
        if (peek() == '}') { pos++; return true; }
        char c = peek();
        if (c == '\n' || c == '\r' || c == '#' || (c == '/' && peek(1) == '/')) {
            consumeBlankLines();
            if (!parseBody()) return false;
            if (peek() != '}') return false;
            pos++;
            return true;
        }
        std::string aid;
        if (!parseIdent(aid)) return false;
        skipWS();
        if (peek() != '=' || peek(1) == '=') return false;
        pos++;
        skipWS();
        if (!parseExpression()) return false;
        skipWS();
        if (peek() != '}') return false;
        pos++;
        return true;
    }

    bool parse() {
        if (src.size() >= 3 && (unsigned char)src[0] == 0xEF && (unsigned char)src[1] == 0xBB && (unsigned char)src[2] == 0xBF) {
            return false;
        }
        if (!parseBody()) return false;
        if (!ok) return false;
        if (!atEnd()) return false;
        return true;
    }
};

int main() {
    std::stringstream ss;
    ss << std::cin.rdbuf();
    std::string input = ss.str();
    Parser p(std::move(input));
    bool valid = p.parse();
    std::cout << (valid ? "valid" : "invalid") << std::endl;
    return 0;
}
