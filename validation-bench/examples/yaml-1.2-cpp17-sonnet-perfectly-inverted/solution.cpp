
#include <iostream>
#include <string>
#include <functional>
#include <cstdint>

// YAML 1.2 syntactic validator

struct Parser {
    std::string s;
    size_t p = 0;

    enum Ctx { FLOW_OUT=0, FLOW_IN=1, BLOCK_KEY=2, FLOW_KEY=3, BLOCK_OUT=4, BLOCK_IN=5 };
    enum Chomp { STRIP, CLIP, KEEP };

    explicit Parser(std::string input) : s(std::move(input)) {}

    bool eof() const { return p >= s.size(); }
    unsigned char cur() const { return eof() ? 0 : (unsigned char)s[p]; }
    unsigned char at_(size_t i) const { return i < s.size() ? (unsigned char)s[i] : 0; }
    bool sol() const { return p == 0 || s[p-1] == '\n'; }

    size_t save() const { return p; }
    void restore(size_t q) { p = q; }

    template<typename F>
    bool attempt(F f) {
        auto q = save();
        if (f()) return true;
        restore(q);
        return false;
    }

    uint32_t peek_cp() const {
        if (eof()) return 0;
        unsigned char c0 = cur();
        if (c0 < 0x80) return c0;
        if (c0 < 0xE0 && p+1 < s.size()) return ((c0&0x1F)<<6)|(at_(p+1)&0x3F);
        if (c0 < 0xF0 && p+2 < s.size()) return ((c0&0x0F)<<12)|((at_(p+1)&0x3F)<<6)|(at_(p+2)&0x3F);
        if (p+3 < s.size()) return ((c0&0x07)<<18)|((at_(p+1)&0x3F)<<12)|((at_(p+2)&0x3F)<<6)|(at_(p+3)&0x3F);
        return c0;
    }
    int cp_len() const {
        unsigned char c0 = cur();
        if (c0 < 0x80) return 1;
        if (c0 < 0xE0) return 2;
        if (c0 < 0xF0) return 3;
        return 4;
    }
    bool advance_cp() { if (eof()) return false; p += cp_len(); return true; }

    bool is_printable(uint32_t c) {
        return c==0x09||c==0x0A||c==0x0D||(c>=0x20&&c<=0x7E)||c==0x85||
               (c>=0xA0&&c<=0xD7FF)||(c>=0xE000&&c<=0xFFFD)||(c>=0x10000&&c<=0x10FFFF);
    }
    bool is_b_char(uint32_t c) { return c==0x0A||c==0x0D; }
    bool is_nb_char(uint32_t c) { return is_printable(c)&&!is_b_char(c)&&c!=0xFEFF; }
    bool is_ns_char(uint32_t c) { return is_nb_char(c)&&c!=0x20&&c!=0x09; }
    bool is_nb_json(uint32_t c) { return c==0x09||c>=0x20; }
    bool is_c_flow_indicator(uint32_t c) { return c==','||c=='['||c==']'||c=='{'||c=='}'; }
    bool is_c_indicator(uint32_t c) {
        return c=='-'||c=='?'||c==':'||c==','||c=='['||c==']'||c=='{'||c=='}'||
               c=='#'||c=='&'||c=='*'||c=='!'||c=='|'||c=='>'||c=='\''||c=='"'||
               c=='%'||c=='@'||c=='`';
    }

    bool b_break() {
        if(cur()=='\r'){p++;if(cur()=='\n')p++;return true;}
        if(cur()=='\n'){p++;return true;}
        return false;
    }
    bool b_non_content() { return b_break(); }
    bool b_as_line_feed() { return b_break(); }
    bool b_as_space() { return b_break(); }

    bool s_space() { if(!eof()&&cur()==0x20){p++;return true;}return false; }
    bool s_tab()   { if(!eof()&&cur()==0x09){p++;return true;}return false; }
    bool s_white() { return s_space()||s_tab(); }
    void while_white() { while(s_white()){} }

    bool nb_char() { uint32_t c=peek_cp(); if(is_nb_char(c)){advance_cp();return true;}return false; }

    bool ns_dec_digit() { return !eof()&&cur()>='0'&&cur()<='9'?(p++,true):false; }
    bool ns_hex_digit() {
        unsigned char c=cur();
        return ((c>='0'&&c<='9')||(c>='A'&&c<='F')||(c>='a'&&c<='f'))?(p++,true):false;
    }
    bool ns_word_char() {
        unsigned char c=cur();
        return ((c>='0'&&c<='9')||(c>='A'&&c<='Z')||(c>='a'&&c<='z')||c=='-')?(p++,true):false;
    }

    bool match_ns_uri_char() {
        if(cur()=='%'){auto q=save();p++;if(ns_hex_digit()&&ns_hex_digit())return true;restore(q);return false;}
        unsigned char c=cur();
        if((c>='0'&&c<='9')||(c>='A'&&c<='Z')||(c>='a'&&c<='z')||c=='-'||
           c=='#'||c==';'||c=='/'||c=='?'||c==':'||c=='@'||c=='&'||c=='='||
           c=='+'||c=='$'||c==','||c=='_'||c=='.'||c=='!'||c=='~'||c=='*'||
           c=='\''||c=='('||c==')'||c=='['||c==']'){p++;return true;}
        return false;
    }
    bool match_ns_tag_char() {
        unsigned char c=cur();
        if(c=='!'||is_c_flow_indicator(c)) return false;
        return match_ns_uri_char();
    }

    bool s_indent(int n) {
        if(n<=0) return true;
        auto q=save();
        for(int i=0;i<n;i++){if(!s_space()){restore(q);return false;}}
        return true;
    }

    int count_leading_spaces() const {
        int cnt=0; size_t i=p;
        while(i<s.size()&&s[i]==' '){cnt++;i++;}
        return cnt;
    }

    bool s_separate_in_line() {
        if(sol()) return true;
        if(!s_white()) return false;
        while_white(); return true;
    }

    bool s_line_prefix(int n, int c) {
        if(!s_indent(n)) return false;
        if(c!=BLOCK_OUT&&c!=BLOCK_IN) attempt([this]{while_white();return true;});
        return true;
    }

    bool s_flow_line_prefix(int n) {
        if(!s_indent(n)) return false;
        attempt([this]{while_white();return true;});
        return true;
    }

    bool l_empty(int n, int c) {
        auto q=save();
        bool ok = attempt([&]{return s_line_prefix(n,c);});
        if(!ok) { int cnt=0; while(!eof()&&cur()==0x20&&cnt<n){p++;cnt++;} ok=true; }
        if(!b_as_line_feed()){restore(q);return false;}
        return true;
    }

    bool b_l_trimmed(int n, int c) {
        auto q=save();
        if(!b_non_content()){restore(q);return false;}
        if(!l_empty(n,c)){restore(q);return false;}
        while(attempt([&]{return l_empty(n,c);})){}
        return true;
    }
    bool b_l_folded(int n, int c) {
        return attempt([&]{return b_l_trimmed(n,c);})||attempt([&]{return b_as_space();});
    }
    bool s_flow_folded(int n) {
        auto q=save();
        attempt([this]{while_white();return true;});
        if(!b_l_folded(n,FLOW_IN)){restore(q);return false;}
        return s_flow_line_prefix(n);
    }

    bool c_nb_comment_text() {
        if(cur()!='#')return false;p++;
        while(nb_char()){}
        return true;
    }
    bool b_comment() { return b_non_content()||eof(); }
    bool s_b_comment() {
        auto q=save();
        attempt([this]{
            if(!s_white())return false;
            while_white();
            attempt([this]{return c_nb_comment_text();});
            return true;
        });
        if(!b_comment()){restore(q);return false;}
        return true;
    }
    bool l_comment() {
        auto q=save();
        if(!s_white()&&!sol()){restore(q);return false;}
        while_white();
        attempt([this]{return c_nb_comment_text();});
        if(!b_comment()){restore(q);return false;}
        return true;
    }
    bool s_l_comments() {
        auto q=save();
        bool ok=attempt([this]{return s_b_comment();})||sol();
        if(!ok){restore(q);return false;}
        while(attempt([this]{return l_comment();})){}
        return true;
    }

    bool s_separate_lines(int n) {
        return attempt([&]{
            if(!s_l_comments())return false;
            return s_flow_line_prefix(n);
        })||attempt([this]{return s_separate_in_line();});
    }
    bool s_separate(int n, int c) {
        if(c==BLOCK_KEY||c==FLOW_KEY) return s_separate_in_line();
        return s_separate_lines(n);
    }

    bool ns_yaml_version() {
        if(!ns_dec_digit())return false;
        while(ns_dec_digit()){}
        if(cur()!='.')return false;p++;
        if(!ns_dec_digit())return false;
        while(ns_dec_digit()){}
        return true;
    }
    bool ns_yaml_directive() {
        if(s.size()-p<4||s.substr(p,4)!="YAML")return false;p+=4;
        if(!s_white())return false;while_white();
        return ns_yaml_version();
    }
    bool c_primary_tag_handle() { if(cur()=='!'){p++;return true;}return false; }
    bool c_secondary_tag_handle() { if(cur()=='!'&&at_(p+1)=='!'){p+=2;return true;}return false; }
    bool c_named_tag_handle() {
        if(cur()!='!')return false;auto q=save();p++;
        if(!ns_word_char()){restore(q);return false;}
        while(ns_word_char()){}
        if(cur()!='!'){restore(q);return false;}p++;return true;
    }
    bool c_tag_handle() {
        return attempt([this]{return c_named_tag_handle();})||
               attempt([this]{return c_secondary_tag_handle();})||
               attempt([this]{return c_primary_tag_handle();});
    }
    bool c_ns_local_tag_prefix() {
        if(cur()!='!')return false;p++;
        while(match_ns_uri_char()){}
        return true;
    }
    bool ns_global_tag_prefix() {
        if(!match_ns_tag_char())return false;
        while(match_ns_uri_char()){}
        return true;
    }
    bool ns_tag_prefix() {
        return attempt([this]{return c_ns_local_tag_prefix();})||
               attempt([this]{return ns_global_tag_prefix();});
    }
    bool ns_tag_directive() {
        if(s.size()-p<3||s.substr(p,3)!="TAG")return false;p+=3;
        if(!s_white())return false;while_white();
        if(!c_tag_handle())return false;
        if(!s_white())return false;while_white();
        return ns_tag_prefix();
    }
    bool ns_reserved_directive() {
        auto q=save();
        uint32_t c=peek_cp();
        if(!is_ns_char(c)){restore(q);return false;}
        advance_cp();
        while(attempt([this]{uint32_t c=peek_cp();if(!is_ns_char(c))return false;advance_cp();return true;})){}
        while(attempt([this]{
            if(!s_white())return false;while_white();
            uint32_t c=peek_cp();if(!is_ns_char(c))return false;advance_cp();
            while(attempt([this]{uint32_t c=peek_cp();if(!is_ns_char(c))return false;advance_cp();return true;})){}
            return true;
        })){}
        return true;
    }
    bool l_directive() {
        auto q=save();
        if(cur()!='%'){restore(q);return false;}p++;
        bool ok=attempt([this]{return ns_yaml_directive();})||
                attempt([this]{return ns_tag_directive();})||
                attempt([this]{return ns_reserved_directive();});
        if(!ok){restore(q);return false;}
        if(!s_l_comments()){restore(q);return false;}
        return true;
    }

    bool c_verbatim_tag() {
        auto q=save();
        if(cur()!='!'||at_(p+1)!='<'){restore(q);return false;}p+=2;
        if(!match_ns_uri_char()){restore(q);return false;}
        while(match_ns_uri_char()){}
        if(cur()!='>'){restore(q);return false;}p++;return true;
    }
    bool c_ns_shorthand_tag() {
        auto q=save();
        if(!c_tag_handle()){restore(q);return false;}
        if(!match_ns_tag_char()){restore(q);return false;}
        while(match_ns_tag_char()){}
        return true;
    }
    bool c_non_specific_tag() { if(cur()=='!'){p++;return true;}return false; }
    bool c_ns_tag_property() {
        return attempt([this]{return c_verbatim_tag();})||
               attempt([this]{return c_ns_shorthand_tag();})||
               attempt([this]{return c_non_specific_tag();});
    }

    bool consume_ns_anchor_char() {
        uint32_t c=peek_cp();
        if(is_ns_char(c)&&!is_c_flow_indicator(c)){advance_cp();return true;}
        return false;
    }
    bool ns_anchor_name() {
        if(!consume_ns_anchor_char())return false;
        while(consume_ns_anchor_char()){}
        return true;
    }
    bool c_ns_anchor_property() { if(cur()!='&')return false;p++;return ns_anchor_name(); }
    bool c_ns_alias_node() { if(cur()!='*')return false;p++;return ns_anchor_name(); }

    bool c_ns_properties(int n, int c) {
        return attempt([&]{
            if(!c_ns_tag_property())return false;
            attempt([&]{if(!s_separate(n,c))return false;return c_ns_anchor_property();});
            return true;
        })||attempt([&]{
            if(!c_ns_anchor_property())return false;
            attempt([&]{if(!s_separate(n,c))return false;return c_ns_tag_property();});
            return true;
        });
    }

    bool c_byte_order_mark() {
        if(p+2<s.size()&&(unsigned char)s[p]==0xEF&&(unsigned char)s[p+1]==0xBB&&(unsigned char)s[p+2]==0xBF){p+=3;return true;}
        return false;
    }

    bool c_ns_esc_char() {
        auto q=save();
        if(cur()!='\\')return false;p++;
        unsigned char c=cur();
        if(c=='0'||c=='a'||c=='b'||c=='t'||c==0x09||c=='n'||c=='v'||c=='f'||
           c=='r'||c=='e'||c==0x20||c=='"'||c=='/'||c=='\\'||c=='N'||c=='_'||c=='L'||c=='P'){p++;return true;}
        if(c=='x'){p++;bool ok=ns_hex_digit()&&ns_hex_digit();if(!ok)restore(q);return ok;}
        if(c=='u'){p++;bool ok=ns_hex_digit()&&ns_hex_digit()&&ns_hex_digit()&&ns_hex_digit();if(!ok)restore(q);return ok;}
        if(c=='U'){p++;bool ok=true;for(int i=0;i<8;i++)if(!ns_hex_digit()){ok=false;break;}if(!ok)restore(q);return ok;}
        restore(q);return false;
    }

    // Double-quoted
    bool nb_double_char() {
        if(attempt([this]{return c_ns_esc_char();}))return true;
        uint32_t c=peek_cp();
        if(is_nb_json(c)&&c!='\\'&&c!='"'){advance_cp();return true;}
        return false;
    }
    bool nb_double_one_line() { while(attempt([this]{return nb_double_char();})){} return true; }
    bool s_double_escaped(int n) {
        auto q=save();
        while_white();
        if(cur()!='\\'){restore(q);return false;}p++;
        if(!b_non_content()){restore(q);return false;}
        while(attempt([&]{return l_empty(n,FLOW_IN);})){}
        return s_flow_line_prefix(n);
    }
    bool s_double_break(int n) {
        return attempt([&]{return s_double_escaped(n);})||attempt([&]{return s_flow_folded(n);});
    }
    bool nb_ns_double_in_line() {
        while(attempt([this]{
            auto q=save();
            while_white();
            uint32_t c=peek_cp();
            if(!is_nb_json(c)||c=='\\'||c=='"'||c==0x20||c==0x09){restore(q);return false;}
            advance_cp();
            return true;
        })){}
        return true;
    }
    bool s_double_next_line(int n) {
        auto q=save();
        if(!s_double_break(n)){restore(q);return false;}
        attempt([&]{
            uint32_t c=peek_cp();
            if(!is_nb_json(c)||c=='\\'||c=='"'||c==0x20||c==0x09)return false;
            advance_cp();
            nb_ns_double_in_line();
            attempt([&]{return s_double_next_line(n);});
            while_white();
            return true;
        });
        return true;
    }
    bool nb_double_multi_line(int n) {
        nb_ns_double_in_line();
        if(!attempt([&]{return s_double_next_line(n);})) while_white();
        return true;
    }
    bool c_double_quoted(int n, int c) {
        auto q=save();
        if(cur()!='"'){restore(q);return false;}p++;
        if(c==BLOCK_KEY||c==FLOW_KEY) nb_double_one_line();
        else nb_double_multi_line(n);
        if(cur()!='"'){restore(q);return false;}p++;
        return true;
    }

    // Single-quoted
    bool c_quoted_quote() { if(cur()=='\''&&at_(p+1)=='\''){p+=2;return true;}return false; }
    bool nb_single_char() {
        if(attempt([this]{return c_quoted_quote();}))return true;
        uint32_t c=peek_cp();
        if(is_nb_json(c)&&c!='\''){advance_cp();return true;}
        return false;
    }
    bool nb_single_one_line() { while(attempt([this]{return nb_single_char();})){} return true; }
    bool s_single_next_line(int n) {
        auto q=save();
        if(!s_flow_folded(n)){restore(q);return false;}
        attempt([&]{
            if(!nb_single_char())return false;
            while(attempt([this]{
                auto q2=save();
                while_white();
                if(!nb_single_char()){restore(q2);return false;}
                return true;
            })){}
            attempt([&]{return s_single_next_line(n);});
            while_white();
            return true;
        });
        return true;
    }
    bool nb_single_multi_line(int n) {
        while(attempt([this]{
            auto q=save();
            while_white();
            if(!nb_single_char()){restore(q);return false;}
            return true;
        })){}
        if(!attempt([&]{return s_single_next_line(n);})) while_white();
        return true;
    }
    bool c_single_quoted(int n, int c) {
        auto q=save();
        if(cur()!='\''){restore(q);return false;}p++;
        if(c==BLOCK_KEY||c==FLOW_KEY) nb_single_one_line();
        else nb_single_multi_line(n);
        if(cur()!='\''){restore(q);return false;}p++;
        return true;
    }

    // Plain
    bool ns_plain_safe(int c) {
        uint32_t ch=peek_cp();
        if(!is_ns_char(ch))return false;
        if(c==FLOW_IN||c==FLOW_KEY) if(is_c_flow_indicator(ch))return false;
        return true;
    }
    bool ns_plain_first(int c) {
        uint32_t ch=peek_cp();
        if(!is_ns_char(ch))return false;
        if(!is_c_indicator(ch)){advance_cp();return true;}
        if(ch==':'||ch=='?'||ch=='-'){
            auto q=save();advance_cp();
            if(ns_plain_safe(c))return true;
            restore(q);
        }
        return false;
    }
    bool ns_plain_char(int c) {
        uint32_t ch=peek_cp();
        if(is_ns_char(ch)&&ch!=':'&&ch!='#'){
            if(c==FLOW_IN||c==FLOW_KEY){if(is_c_flow_indicator(ch))return false;}
            advance_cp();return true;
        }
        if(ch=='#'&&p>0){
            unsigned char pb=(unsigned char)s[p-1];
            if(is_ns_char(pb)){advance_cp();return true;}
        }
        if(ch==':'){
            auto q=save();advance_cp();
            if(ns_plain_safe(c))return true;
            restore(q);
        }
        return false;
    }
    bool nb_ns_plain_in_line(int c) {
        while(attempt([&]{
            auto q=save();
            while_white();
            if(!ns_plain_char(c)){restore(q);return false;}
            return true;
        })){}
        return true;
    }
    bool ns_plain_one_line(int c) {
        if(!ns_plain_first(c))return false;
        nb_ns_plain_in_line(c);
        return true;
    }
    bool s_ns_plain_next_line(int n, int c) {
        auto q=save();
        if(!s_flow_folded(n)){restore(q);return false;}
        if(!ns_plain_char(c)){restore(q);return false;}
        nb_ns_plain_in_line(c);
        return true;
    }
    bool ns_plain_multi_line(int n, int c) {
        if(!ns_plain_one_line(c))return false;
        while(attempt([&]{return s_ns_plain_next_line(n,c);})){}
        return true;
    }
    bool ns_plain(int n, int c) {
        if(c==BLOCK_KEY||c==FLOW_KEY) return ns_plain_one_line(c);
        return ns_plain_multi_line(n,c);
    }

    // Block scalars
    bool c_indentation_indicator(int& m) {
        if(cur()>='1'&&cur()<='9'){m=cur()-'0';p++;return true;}
        m=0;return false;
    }
    bool c_chomping_indicator(Chomp& t) {
        if(cur()=='-'){t=STRIP;p++;return true;}
        if(cur()=='+'){t=KEEP;p++;return true;}
        t=CLIP;return false;
    }
    bool c_b_block_header(int& m, Chomp& t) {
        auto q=save();
        bool gi=c_indentation_indicator(m);
        bool gc=c_chomping_indicator(t);
        if(!gi) c_indentation_indicator(m);
        if(!gc) c_chomping_indicator(t);
        if(!s_b_comment()){restore(q);return false;}
        return true;
    }

    bool l_trail_comments(int n) {
        auto q=save();
        int cnt=0;
        while(!eof()&&cur()==' '&&cnt<n){p++;cnt++;}
        if(cur()!='#'){restore(q);return false;}
        if(!c_nb_comment_text()){restore(q);return false;}
        if(!b_comment()){restore(q);return false;}
        while(attempt([this]{return l_comment();})){}
        return true;
    }
    bool l_strip_empty(int n) {
        while(attempt([&]{
            auto q=save();
            int cnt=0;
            while(!eof()&&cur()==' '&&cnt<=n){p++;cnt++;}
            if(!b_non_content()){restore(q);return false;}
            return true;
        })){}
        attempt([&]{return l_trail_comments(n);});
        return true;
    }
    bool l_keep_empty(int n) {
        while(attempt([&]{return l_empty(n,BLOCK_IN);})){}
        attempt([&]{return l_trail_comments(n);});
        return true;
    }
    bool l_chomped_empty(int n, Chomp t) {
        return (t==KEEP)?l_keep_empty(n):l_strip_empty(n);
    }
    bool b_chomped_last(Chomp t) {
        if(t==STRIP) return b_non_content()||eof();
        return b_as_line_feed()||eof();
    }

    bool l_nb_literal_text(int n) {
        auto q=save();
        while(attempt([&]{return l_empty(n,BLOCK_IN);})){}
        if(!s_indent(n)){restore(q);return false;}
        if(!nb_char()){restore(q);return false;}
        while(nb_char()){}
        return true;
    }
    bool b_nb_literal_next(int n) {
        auto q=save();
        if(!b_as_line_feed()){restore(q);return false;}
        if(!l_nb_literal_text(n)){restore(q);return false;}
        return true;
    }
    bool l_literal_content(int n, Chomp t) {
        attempt([&]{
            if(!l_nb_literal_text(n))return false;
            while(attempt([&]{return b_nb_literal_next(n);})){}
            return b_chomped_last(t);
        });
        l_chomped_empty(n,t);
        return true;
    }
    int detect_block_indent(int n) {
        // Look ahead to find first non-empty line and count its leading spaces
        auto q=save();
        while(!eof()){
            int ai=count_leading_spaces();
            for(int i=0;i<ai;i++) p++;
            if(!eof()&&(cur()=='\n'||cur()=='\r')){b_break();continue;}
            break;
        }
        int m=count_leading_spaces()-n;
        restore(q);
        return m<1?1:m;
    }
    bool c_l_plus_literal(int n) {
        auto q=save();
        if(cur()!='|'){restore(q);return false;}p++;
        int m=0;Chomp t=CLIP;
        if(!c_b_block_header(m,t)){restore(q);return false;}
        if(m==0) m=detect_block_indent(n);
        return l_literal_content(n+m,t);
    }

    bool s_nb_folded_text(int n) {
        auto q=save();
        if(!s_indent(n)){restore(q);return false;}
        uint32_t c=peek_cp();
        if(!is_ns_char(c)){restore(q);return false;}
        advance_cp();
        while(nb_char()){}
        return true;
    }
    bool l_nb_folded_lines(int n) {
        auto q=save();
        if(!s_nb_folded_text(n)){restore(q);return false;}
        while(attempt([&]{
            if(!b_l_folded(n,BLOCK_IN))return false;
            return s_nb_folded_text(n);
        })){}
        return true;
    }
    bool s_nb_spaced_text(int n) {
        auto q=save();
        if(!s_indent(n)){restore(q);return false;}
        if(!s_white()){restore(q);return false;}
        while(nb_char()){}
        return true;
    }
    bool b_l_spaced(int n) {
        auto q=save();
        if(!b_as_line_feed()){restore(q);return false;}
        while(attempt([&]{return l_empty(n,BLOCK_IN);})){}
        return true;
    }
    bool l_nb_spaced_lines(int n) {
        auto q=save();
        if(!s_nb_spaced_text(n)){restore(q);return false;}
        while(attempt([&]{
            if(!b_l_spaced(n))return false;
            return s_nb_spaced_text(n);
        })){}
        return true;
    }
    bool l_nb_same_lines(int n) {
        while(attempt([&]{return l_empty(n,BLOCK_IN);})){}
        return attempt([&]{return l_nb_folded_lines(n);})||
               attempt([&]{return l_nb_spaced_lines(n);});
    }
    bool l_nb_diff_lines(int n) {
        if(!l_nb_same_lines(n))return false;
        while(attempt([&]{
            if(!b_as_line_feed())return false;
            return l_nb_same_lines(n);
        })){}
        return true;
    }
    bool l_folded_content(int n, Chomp t) {
        attempt([&]{
            if(!l_nb_diff_lines(n))return false;
            return b_chomped_last(t);
        });
        l_chomped_empty(n,t);
        return true;
    }
    bool c_l_plus_folded(int n) {
        auto q=save();
        if(cur()!='>'){restore(q);return false;}p++;
        int m=0;Chomp t=CLIP;
        if(!c_b_block_header(m,t)){restore(q);return false;}
        if(m==0) m=detect_block_indent(n);
        return l_folded_content(n+m,t);
    }

    // Flow collections - uses ns_flow_node which is defined later via a dispatch
    // We use std::function to break the circular dependency
    std::function<bool(int,int)> flow_node_fn;
    std::function<bool(int,int)> flow_json_node_fn;
    std::function<bool(int,int)> flow_yaml_node_fn;

    int in_flow_ctx(int c) {
        return (c==FLOW_OUT||c==FLOW_IN)?FLOW_IN:FLOW_KEY;
    }

    bool c_ns_flow_map_separate_value(int n, int c) {
        auto q=save();
        if(cur()!=':'){restore(q);return false;}p++;
        if(ns_plain_safe(c)){restore(q);return false;}
        attempt([&]{
            if(!s_separate(n,c))return false;
            return flow_node_fn(n,c);
        });
        return true;
    }
    bool c_ns_flow_map_adjacent_value(int n, int c) {
        auto q=save();
        if(cur()!=':'){restore(q);return false;}p++;
        attempt([&]{
            attempt([&]{return s_separate(n,c);});
            return flow_node_fn(n,c);
        });
        return true;
    }
    bool ns_flow_map_yaml_key_entry(int n, int c) {
        auto q=save();
        if(!flow_yaml_node_fn(n,c)){restore(q);return false;}
        attempt([&]{
            attempt([&]{return s_separate(n,c);});
            return c_ns_flow_map_separate_value(n,c);
        });
        return true;
    }
    bool c_ns_flow_map_empty_key_entry(int n, int c) {
        return c_ns_flow_map_separate_value(n,c);
    }
    bool c_ns_flow_map_json_key_entry(int n, int c) {
        auto q=save();
        if(!flow_json_node_fn(n,c)){restore(q);return false;}
        attempt([&]{
            attempt([&]{return s_separate(n,c);});
            return c_ns_flow_map_adjacent_value(n,c);
        });
        return true;
    }
    bool ns_flow_map_implicit_entry(int n, int c) {
        return attempt([&]{return ns_flow_map_yaml_key_entry(n,c);})||
               attempt([&]{return c_ns_flow_map_empty_key_entry(n,c);})||
               attempt([&]{return c_ns_flow_map_json_key_entry(n,c);});
    }
    bool ns_flow_map_explicit_entry(int n, int c) {
        if(attempt([&]{return ns_flow_map_implicit_entry(n,c);}))return true;
        return true; // e-node e-node
    }
    bool ns_flow_map_entry(int n, int c) {
        return attempt([&]{
            if(cur()!='?')return false;
            auto q=save();p++;
            if(!eof()&&is_ns_char(peek_cp())){restore(q);return false;}
            if(!s_separate(n,c)){restore(q);return false;}
            return ns_flow_map_explicit_entry(n,c);
        })||attempt([&]{return ns_flow_map_implicit_entry(n,c);});
    }
    bool ns_s_flow_map_entries(int n, int c) {
        auto q=save();
        if(!ns_flow_map_entry(n,c)){restore(q);return false;}
        attempt([&]{return s_separate(n,c);});
        attempt([&]{
            if(cur()!=',')return false;p++;
            attempt([&]{return s_separate(n,c);});
            attempt([&]{return ns_s_flow_map_entries(n,c);});
            return true;
        });
        return true;
    }
    bool c_flow_mapping(int n, int c) {
        auto q=save();
        if(cur()!='{'){restore(q);return false;}p++;
        attempt([&]{return s_separate(n,c);});
        attempt([&]{return ns_s_flow_map_entries(n,in_flow_ctx(c));});
        if(cur()!='}'){restore(q);return false;}p++;
        return true;
    }

    bool ns_s_implicit_yaml_key(int c) {
        auto q=save();
        if(!flow_yaml_node_fn(0,c)){restore(q);return false;}
        attempt([this]{return s_separate_in_line();});
        return true;
    }
    bool c_s_implicit_json_key(int c) {
        auto q=save();
        if(!flow_json_node_fn(0,c)){restore(q);return false;}
        attempt([this]{return s_separate_in_line();});
        return true;
    }
    bool ns_flow_pair_yaml_key_entry(int n, int c) {
        auto q=save();
        if(!ns_s_implicit_yaml_key(FLOW_KEY)){restore(q);return false;}
        return c_ns_flow_map_separate_value(n,c);
    }
    bool c_ns_flow_pair_json_key_entry(int n, int c) {
        auto q=save();
        if(!c_s_implicit_json_key(FLOW_KEY)){restore(q);return false;}
        return c_ns_flow_map_adjacent_value(n,c);
    }
    bool ns_flow_pair_entry(int n, int c) {
        return attempt([&]{return ns_flow_pair_yaml_key_entry(n,c);})||
               attempt([&]{return c_ns_flow_map_empty_key_entry(n,c);})||
               attempt([&]{return c_ns_flow_pair_json_key_entry(n,c);});
    }
    bool ns_flow_pair(int n, int c) {
        return attempt([&]{
            if(cur()!='?')return false;
            auto q=save();p++;
            if(!eof()&&is_ns_char(peek_cp())){restore(q);return false;}
            if(!s_separate(n,c)){restore(q);return false;}
            return ns_flow_map_explicit_entry(n,c);
        })||attempt([&]{return ns_flow_pair_entry(n,c);});
    }

    bool ns_flow_seq_entry(int n, int c) {
        return attempt([&]{return ns_flow_pair(n,c);})||
               attempt([&]{return flow_node_fn(n,c);});
    }
    bool ns_s_flow_seq_entries(int n, int c) {
        auto q=save();
        if(!ns_flow_seq_entry(n,c)){restore(q);return false;}
        attempt([&]{return s_separate(n,c);});
        attempt([&]{
            if(cur()!=',')return false;p++;
            attempt([&]{return s_separate(n,c);});
            attempt([&]{return ns_s_flow_seq_entries(n,c);});
            return true;
        });
        return true;
    }
    bool c_flow_sequence(int n, int c) {
        auto q=save();
        if(cur()!='['){restore(q);return false;}p++;
        attempt([&]{return s_separate(n,c);});
        attempt([&]{return ns_s_flow_seq_entries(n,in_flow_ctx(c));});
        if(cur()!=']'){restore(q);return false;}p++;
        return true;
    }

    bool c_flow_json_content(int n, int c) {
        return attempt([&]{return c_flow_sequence(n,c);})||
               attempt([&]{return c_flow_mapping(n,c);})||
               attempt([&]{return c_single_quoted(n,c);})||
               attempt([&]{return c_double_quoted(n,c);});
    }
    bool ns_flow_content(int n, int c) {
        return attempt([&]{return ns_plain(n,c);})||
               attempt([&]{return c_flow_json_content(n,c);});
    }

    // These are the actual implementations called via std::function
    bool ns_flow_yaml_node_impl(int n, int c) {
        if(attempt([this]{return c_ns_alias_node();}))return true;
        if(attempt([&]{return ns_plain(n,c);}))return true;
        return attempt([&]{
            if(!c_ns_properties(n,c))return false;
            if(!attempt([&]{
                if(!s_separate(n,c))return false;
                return ns_plain(n,c);
            })){}
            return true;
        });
    }
    bool c_flow_json_node_impl(int n, int c) {
        attempt([&]{
            if(!c_ns_properties(n,c))return false;
            return s_separate(n,c);
        });
        return c_flow_json_content(n,c);
    }
    bool ns_flow_node_impl(int n, int c) {
        if(attempt([this]{return c_ns_alias_node();}))return true;
        if(attempt([&]{return ns_flow_content(n,c);}))return true;
        return attempt([&]{
            if(!c_ns_properties(n,c))return false;
            if(!attempt([&]{
                if(!s_separate(n,c))return false;
                return ns_flow_content(n,c);
            })){}
            return true;
        });
    }

    // Block collections
    bool s_l_plus_block_node_impl(int n, int c);

    bool ns_l_compact_sequence(int n) {
        auto q=save();
        if(!c_l_block_seq_entry(n)){restore(q);return false;}
        while(attempt([&]{if(!s_indent(n))return false;return c_l_block_seq_entry(n);})){}
        return true;
    }
    bool ns_l_compact_mapping(int n) {
        auto q=save();
        if(!ns_l_block_map_entry(n)){restore(q);return false;}
        while(attempt([&]{if(!s_indent(n))return false;return ns_l_block_map_entry(n);})){}
        return true;
    }
    bool s_l_plus_block_indented(int n, int c) {
        int m_spaces=count_leading_spaces();
        if(m_spaces>0){
            if(attempt([&]{
                auto q=save();
                if(!s_indent(m_spaces)){restore(q);return false;}
                if(!ns_l_compact_sequence(n+1+m_spaces)){restore(q);return false;}
                return true;
            }))return true;
            if(attempt([&]{
                auto q=save();
                if(!s_indent(m_spaces)){restore(q);return false;}
                if(!ns_l_compact_mapping(n+1+m_spaces)){restore(q);return false;}
                return true;
            }))return true;
        }
        if(attempt([&]{return s_l_plus_block_node_impl(n,c);}))return true;
        return s_l_comments();
    }
    bool c_l_block_seq_entry(int n) {
        auto q=save();
        if(cur()!='-'){restore(q);return false;}p++;
        if(!eof()&&is_ns_char(peek_cp())){restore(q);return false;}
        if(!s_l_plus_block_indented(n,BLOCK_IN)){restore(q);return false;}
        return true;
    }
    bool l_plus_block_sequence(int n) {
        auto q=save();
        int ai=count_leading_spaces();
        if(ai<=n){restore(q);return false;}
        int new_n=ai;
        if(!s_indent(new_n)){restore(q);return false;}
        if(!c_l_block_seq_entry(new_n)){restore(q);return false;}
        while(attempt([&]{if(!s_indent(new_n))return false;return c_l_block_seq_entry(new_n);})){}
        return true;
    }
    bool c_l_block_map_explicit_key(int n) {
        auto q=save();
        if(cur()!='?'){restore(q);return false;}p++;
        if(!eof()&&is_ns_char(peek_cp())){restore(q);return false;}
        if(!s_l_plus_block_indented(n,BLOCK_OUT)){restore(q);return false;}
        return true;
    }
    bool l_block_map_explicit_value(int n) {
        auto q=save();
        if(!s_indent(n)){restore(q);return false;}
        if(cur()!=':'){restore(q);return false;}p++;
        if(!eof()&&is_ns_char(peek_cp())){restore(q);return false;}
        if(!s_l_plus_block_indented(n,BLOCK_OUT)){restore(q);return false;}
        return true;
    }
    bool c_l_block_map_explicit_entry(int n) {
        auto q=save();
        if(!c_l_block_map_explicit_key(n)){restore(q);return false;}
        attempt([&]{return l_block_map_explicit_value(n);});
        return true;
    }
    bool ns_s_block_map_implicit_key() {
        return attempt([this]{return c_s_implicit_json_key(BLOCK_KEY);})||
               attempt([this]{return ns_s_implicit_yaml_key(BLOCK_KEY);});
    }
    bool c_l_block_map_implicit_value(int n) {
        auto q=save();
        if(cur()!=':'){restore(q);return false;}p++;
        if(!eof()&&is_ns_char(peek_cp())){restore(q);return false;}
        if(!attempt([&]{return s_l_plus_block_node_impl(n,BLOCK_OUT);})){
            s_l_comments();
        }
        return true;
    }
    bool ns_l_block_map_implicit_entry(int n) {
        auto q=save();
        attempt([this]{return ns_s_block_map_implicit_key();});
        if(!c_l_block_map_implicit_value(n)){restore(q);return false;}
        return true;
    }
    bool ns_l_block_map_entry(int n) {
        return attempt([&]{return c_l_block_map_explicit_entry(n);})||
               attempt([&]{return ns_l_block_map_implicit_entry(n);});
    }
    bool l_plus_block_mapping(int n) {
        auto q=save();
        int ai=count_leading_spaces();
        if(ai<=n){restore(q);return false;}
        int new_n=ai;
        if(!s_indent(new_n)){restore(q);return false;}
        if(!ns_l_block_map_entry(new_n)){restore(q);return false;}
        while(attempt([&]{if(!s_indent(new_n))return false;return ns_l_block_map_entry(new_n);})){}
        return true;
    }
    bool s_l_plus_block_scalar(int n, int c) {
        auto q=save();
        if(!s_separate(n+1,c)){restore(q);return false;}
        attempt([&]{
            if(!c_ns_properties(n+1,c))return false;
            return s_separate(n+1,c);
        });
        return attempt([&]{return c_l_plus_literal(n);})||
               attempt([&]{return c_l_plus_folded(n);});
    }
    bool seq_space(int n, int c) {
        return (c==BLOCK_OUT)?l_plus_block_sequence(n-1):l_plus_block_sequence(n);
    }
    bool s_l_plus_block_collection(int n, int c) {
        auto q=save();
        attempt([&]{
            if(!s_separate(n+1,c))return false;
            return c_ns_properties(n+1,c);
        });
        if(!s_l_comments()){restore(q);return false;}
        return attempt([&]{return seq_space(n,c);})||
               attempt([&]{return l_plus_block_mapping(n);});
    }
    bool s_l_plus_block_in_block(int n, int c) {
        return attempt([&]{return s_l_plus_block_scalar(n,c);})||
               attempt([&]{return s_l_plus_block_collection(n,c);});
    }
    bool s_l_plus_flow_in_block(int n) {
        auto q=save();
        if(!s_separate(n+1,FLOW_OUT)){restore(q);return false;}
        if(!flow_node_fn(n+1,FLOW_OUT)){restore(q);return false;}
        if(!s_l_comments()){restore(q);return false;}
        return true;
    }

    // Documents
    bool l_document_prefix() {
        attempt([this]{return c_byte_order_mark();});
        while(attempt([this]{return l_comment();})){}
        return true;
    }
    bool c_directives_end() {
        auto q=save();
        if(s.size()-p<3||s.substr(p,3)!="---"){restore(q);return false;}
        p+=3;
        if(eof()||is_b_char(cur())||cur()==' '||cur()=='\t'||cur()=='#')return true;
        restore(q);return false;
    }
    bool c_document_end() {
        auto q=save();
        if(s.size()-p<3||s.substr(p,3)!="..."){restore(q);return false;}
        p+=3;
        if(eof()||is_b_char(cur())||cur()==' '||cur()=='\t')return true;
        restore(q);return false;
    }
    bool l_document_suffix() {
        auto q=save();
        if(!c_document_end()){restore(q);return false;}
        if(!s_l_comments()){restore(q);return false;}
        return true;
    }
    bool l_bare_document() {
        return attempt([this]{return s_l_plus_block_node_impl(-1,BLOCK_IN);});
    }
    bool l_explicit_document() {
        auto q=save();
        if(!c_directives_end()){restore(q);return false;}
        if(!attempt([this]{return l_bare_document();})) s_l_comments();
        return true;
    }
    bool l_directive_document() {
        auto q=save();
        if(!l_directive()){restore(q);return false;}
        while(attempt([this]{return l_directive();})){}
        return l_explicit_document();
    }
    bool l_any_document() {
        return attempt([this]{return l_directive_document();})||
               attempt([this]{return l_explicit_document();})||
               attempt([this]{return l_bare_document();});
    }
    bool l_yaml_stream() {
        // Set up function pointers for mutual recursion
        flow_node_fn = [this](int n, int c){ return ns_flow_node_impl(n,c); };
        flow_json_node_fn = [this](int n, int c){ return c_flow_json_node_impl(n,c); };
        flow_yaml_node_fn = [this](int n, int c){ return ns_flow_yaml_node_impl(n,c); };

        while(attempt([this]{return l_document_prefix();})){}
        attempt([this]{return l_any_document();});
        while(true){
            bool found=false;
            found|=attempt([this]{
                if(!l_document_suffix())return false;
                while(attempt([this]{return l_document_suffix();})){}
                while(attempt([this]{return l_document_prefix();})){}
                attempt([this]{return l_any_document();});
                return true;
            });
            if(!found)found|=attempt([this]{return c_byte_order_mark();});
            if(!found)found|=attempt([this]{return l_comment();});
            if(!found)found|=attempt([this]{return l_explicit_document();});
            if(!found)break;
        }
        return eof();
    }
    bool parse(){return l_yaml_stream();}
};

bool Parser::s_l_plus_block_node_impl(int n, int c) {
    return attempt([&]{return s_l_plus_block_in_block(n,c);})||
           attempt([&]{return s_l_plus_flow_in_block(n);});
}

int main(){
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::string input,line;
    while(std::getline(std::cin,line)){input+=line+"\n";}
    Parser parser(input);
    std::cout<<(parser.parse()?"valid":"invalid")<<std::endl;
    return 0;
}
