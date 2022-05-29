use logos::Logos;
use std::io::{self, Read};

fn main() -> io::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    let mut tokens = Vec::new();
    let mut lexer = TokenKind::lexer(&input);
    while let Some(kind) = lexer.next() {
        tokens.push(Token {
            kind,
            text: lexer.slice().to_string(),
            pos: lexer.span().start,
        });
    }

    let parser = Parser { tokens, cursor: 0 };
    let ast = parser.parse();

    println!("{}", ast.codegen());

    Ok(())
}

#[derive(Debug)]
struct Token {
    kind: TokenKind,
    text: String,
    pos: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Logos)]
enum TokenKind {
    #[token("fn")]
    FnKw,

    #[token("struct")]
    StructKw,

    #[token("let")]
    LetKw,

    #[token("if")]
    IfKw,

    #[token("else")]
    ElseKw,

    #[token("for")]
    ForKw,

    #[token("return")]
    ReturnKw,

    #[token("break")]
    BreakKw,

    #[token("char")]
    CharKw,

    #[token("int")]
    IntKw,

    #[regex("[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,

    #[regex("[0-9]+")]
    Number,

    #[regex("'[^']+'")]
    Char,

    #[regex(r#""([^"]|\\")*""#)]
    String,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token(".")]
    Dot,

    #[token(":")]
    Colon,

    #[token(";")]
    Semi,

    #[token("!")]
    Bang,

    #[token(",")]
    Comma,

    #[token("&")]
    And,

    #[token("=")]
    Eq,

    #[token("+")]
    Plus,

    #[token("-")]
    Hyphen,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token("<")]
    Lt,

    #[token(">")]
    Gt,

    #[token("|")]
    Pipe,

    #[token("->")]
    Arrow,

    #[regex("(\\s+|#.*)", logos::skip)]
    #[error]
    Error,
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
}

impl Parser {
    fn parse(mut self) -> Ast {
        let mut items = Vec::new();

        while !self.at_eof() {
            items.push(self.parse_item());
        }

        Ast(items)
    }

    fn parse_item(&mut self) -> Item {
        match self.bump() {
            TokenKind::FnKw => {
                let name = self.expect(TokenKind::Ident);

                self.expect(TokenKind::LParen);

                let mut params = Vec::new();
                while self.peek() != TokenKind::RParen {
                    let name = self.expect(TokenKind::Ident);
                    self.expect(TokenKind::Colon);
                    let ty = self.parse_ty();
                    params.push((name, ty));

                    if self.peek() == TokenKind::Comma {
                        self.bump();
                    }
                }

                self.expect(TokenKind::RParen);

                let return_ty = if self.peek() == TokenKind::LBrace {
                    None
                } else {
                    Some(self.parse_ty())
                };

                self.expect(TokenKind::LBrace);

                let mut body = Vec::new();
                while self.peek() != TokenKind::RBrace {
                    body.push(self.parse_statement());
                }

                self.expect(TokenKind::RBrace);

                Item::Function(Function {
                    name,
                    params,
                    return_ty,
                    body,
                })
            }

            TokenKind::StructKw => {
                let name = self.expect(TokenKind::Ident);

                let mut fields = Vec::new();

                self.expect(TokenKind::LBrace);
                while self.peek() != TokenKind::RBrace {
                    let name = self.expect(TokenKind::Ident);
                    self.expect(TokenKind::Colon);
                    let ty = self.parse_ty();

                    if self.peek() == TokenKind::Comma {
                        self.bump();
                    }

                    fields.push((name, ty));
                }
                self.expect(TokenKind::RBrace);

                Item::Struct { name, fields }
            }

            _ => self.error("item"),
        }
    }

    fn parse_statement(&mut self) -> Statement {
        match self.peek() {
            TokenKind::LetKw => {
                self.bump();
                let name = self.expect(TokenKind::Ident);
                self.expect(TokenKind::Colon);
                let ty = self.parse_ty();

                let val = if self.peek() == TokenKind::Eq {
                    self.expect(TokenKind::Eq);
                    Some(self.parse_expr())
                } else {
                    None
                };

                self.expect(TokenKind::Semi);

                Statement::Let { name, ty, val }
            }

            TokenKind::IfKw => {
                self.bump();
                let condition = self.parse_expr();

                self.expect(TokenKind::LBrace);
                let mut true_branch = Vec::new();
                while self.peek() != TokenKind::RBrace {
                    true_branch.push(self.parse_statement());
                }
                self.expect(TokenKind::RBrace);

                let false_branch = if self.peek() == TokenKind::ElseKw {
                    self.bump();
                    self.expect(TokenKind::LBrace);
                    let mut false_branch = Vec::new();
                    while self.peek() != TokenKind::RBrace {
                        false_branch.push(self.parse_statement());
                    }
                    self.expect(TokenKind::RBrace);

                    Some(false_branch)
                } else {
                    None
                };

                Statement::If {
                    condition,
                    true_branch,
                    false_branch,
                }
            }

            TokenKind::ForKw => {
                self.bump();

                self.expect(TokenKind::LBrace);
                let mut body = Vec::new();
                while self.peek() != TokenKind::RBrace {
                    body.push(self.parse_statement());
                }
                self.expect(TokenKind::RBrace);

                Statement::Loop { body }
            }

            TokenKind::BreakKw => {
                self.bump();
                self.expect(TokenKind::Semi);
                Statement::Break
            }

            TokenKind::ReturnKw => {
                self.bump();
                let val = if self.peek() == TokenKind::Semi {
                    None
                } else {
                    Some(self.parse_expr())
                };
                self.expect(TokenKind::Semi);
                Statement::Return { val }
            }

            _ => {
                let e = self.parse_expr();
                self.expect(TokenKind::Semi);
                Statement::Expr(e)
            }
        }
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Expr {
        let mut lhs = self.parse_lhs();

        loop {
            let (op, num_op_tokens) = match (self.peek(), self.lookahead()) {
                (TokenKind::Eq, TokenKind::Eq) => (BinaryOp::Eq, 2),
                (TokenKind::Bang, TokenKind::Eq) => (BinaryOp::NEq, 2),
                (TokenKind::Eq, _) => (BinaryOp::Assign, 1),
                (TokenKind::Plus, TokenKind::Eq) => (BinaryOp::AddAssign, 2),
                (TokenKind::Hyphen, TokenKind::Eq) => (BinaryOp::SubAssign, 2),
                (TokenKind::Star, TokenKind::Eq) => (BinaryOp::MulAssign, 2),
                (TokenKind::Slash, TokenKind::Eq) => (BinaryOp::DivAssign, 2),
                (TokenKind::Plus, _) => (BinaryOp::Add, 1),
                (TokenKind::Hyphen, _) => (BinaryOp::Sub, 1),
                (TokenKind::Star, _) => (BinaryOp::Mul, 1),
                (TokenKind::Slash, _) => (BinaryOp::Div, 1),
                (TokenKind::Lt, TokenKind::Eq) => (BinaryOp::LtEq, 2),
                (TokenKind::Gt, TokenKind::Eq) => (BinaryOp::GtEq, 2),
                (TokenKind::Lt, _) => (BinaryOp::Lt, 1),
                (TokenKind::Gt, _) => (BinaryOp::Gt, 1),
                (TokenKind::And, TokenKind::And) => (BinaryOp::And, 2),
                (TokenKind::Pipe, TokenKind::Pipe) => (BinaryOp::Or, 2),
                _ => break,
            };

            let (l_bp, r_bp) = op.bp();
            if l_bp < min_bp {
                break;
            }

            for _ in 0..num_op_tokens {
                self.bump();
            }

            let rhs = self.parse_expr_bp(r_bp);
            lhs = Expr::Binary {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            };
        }

        lhs
    }

    fn parse_lhs(&mut self) -> Expr {
        let mut e = match self.bump() {
            TokenKind::Ident => {
                let ident = self.tokens[self.cursor - 1].text.clone();
                if self.peek() == TokenKind::LParen {
                    self.bump();

                    let mut args = Vec::new();
                    while self.peek() != TokenKind::RParen {
                        args.push(self.parse_expr());
                        if self.peek() == TokenKind::Comma {
                            self.bump();
                        }
                    }
                    self.expect(TokenKind::RParen);

                    Expr::Call {
                        function: ident,
                        args,
                    }
                } else {
                    Expr::Variable(ident)
                }
            }
            TokenKind::Number => Expr::Int(self.tokens[self.cursor - 1].text.parse().unwrap()),
            TokenKind::Char => {
                let text = &self.tokens[self.cursor - 1].text;
                Expr::Char(text[1..text.len() - 1].to_string())
            }
            TokenKind::String => {
                let text = &self.tokens[self.cursor - 1].text;
                Expr::String(text[1..text.len() - 1].to_string())
            }
            TokenKind::LParen => {
                let e = self.parse_expr();
                self.expect(TokenKind::RParen);
                e
            }
            TokenKind::Star => Expr::Unary {
                op: UnaryOp::Dereference,
                expr: Box::new(self.parse_expr_bp(255)),
            },
            TokenKind::And => Expr::Unary {
                op: UnaryOp::TakeAddress,
                expr: Box::new(self.parse_expr_bp(255)),
            },
            TokenKind::Bang => Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(self.parse_expr_bp(255)),
            },
            _ => self.error("expression"),
        };

        loop {
            match self.peek() {
                TokenKind::Dot => {
                    self.bump();
                    let field = self.expect(TokenKind::Ident);
                    e = Expr::FieldAccess {
                        expr: Box::new(e),
                        field,
                        through_pointer: false,
                    };
                }
                TokenKind::Arrow => {
                    self.bump();
                    let field = self.expect(TokenKind::Ident);
                    e = Expr::FieldAccess {
                        expr: Box::new(e),
                        field,
                        through_pointer: true,
                    };
                }
                _ => break,
            }
        }

        e
    }

    fn parse_ty(&mut self) -> Ty {
        match self.bump() {
            TokenKind::CharKw => Ty::Char,
            TokenKind::IntKw => Ty::Int,
            TokenKind::Ident => Ty::Named(self.tokens[self.cursor - 1].text.clone()),
            TokenKind::And => Ty::Pointer(Box::new(self.parse_ty())),
            _ => self.error("type"),
        }
    }

    fn expect(&mut self, k: TokenKind) -> String {
        if self.peek() != k {
            self.error(&format!("{k:?}"));
        }

        let text = self.tokens[self.cursor].text.clone();
        self.cursor += 1;

        text
    }

    fn error(&self, expected: &str) -> ! {
        panic!("expected {expected} at {}", self.tokens[self.cursor].pos)
    }

    fn bump(&mut self) -> TokenKind {
        let k = self.peek();
        self.cursor += 1;
        k
    }

    fn peek(&self) -> TokenKind {
        self.tokens[self.cursor].kind
    }

    fn lookahead(&self) -> TokenKind {
        self.tokens[self.cursor + 1].kind
    }

    fn at_eof(&self) -> bool {
        self.cursor >= self.tokens.len()
    }
}

#[derive(Debug)]
struct Ast(Vec<Item>);

#[derive(Debug)]
enum Item {
    Function(Function),
    Struct {
        name: String,
        fields: Vec<(String, Ty)>,
    },
}

#[derive(Debug)]
struct Function {
    name: String,
    params: Vec<(String, Ty)>,
    return_ty: Option<Ty>,
    body: Vec<Statement>,
}

#[derive(Debug)]
enum Statement {
    Let {
        name: String,
        ty: Ty,
        val: Option<Expr>,
    },
    Return {
        val: Option<Expr>,
    },
    Break,
    If {
        condition: Expr,
        true_branch: Vec<Statement>,
        false_branch: Option<Vec<Statement>>,
    },
    Loop {
        body: Vec<Statement>,
    },
    Expr(Expr),
}

#[derive(Debug)]
enum Expr {
    Variable(String),
    Int(u32),
    Char(String),
    String(String),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        lhs: Box<Expr>,
        op: BinaryOp,
        rhs: Box<Expr>,
    },
    Call {
        function: String,
        args: Vec<Expr>,
    },
    FieldAccess {
        expr: Box<Expr>,
        field: String,
        through_pointer: bool,
    },
}

#[derive(Debug, Clone, Copy)]
enum UnaryOp {
    Not,
    TakeAddress,
    Dereference,
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    Eq,
    NEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
}

impl BinaryOp {
    fn bp(self) -> (u8, u8) {
        match self {
            BinaryOp::Add | BinaryOp::Sub => (11, 12),
            BinaryOp::Mul | BinaryOp::Div => (13, 14),
            BinaryOp::Assign
            | BinaryOp::AddAssign
            | BinaryOp::SubAssign
            | BinaryOp::MulAssign
            | BinaryOp::DivAssign => (1, 2),
            BinaryOp::Eq | BinaryOp::NEq => (7, 8),
            BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => (9, 10),
            BinaryOp::And => (5, 6),
            BinaryOp::Or => (3, 4),
        }
    }
}

#[derive(Debug)]
enum Ty {
    Char,
    Int,
    Named(String),
    Pointer(Box<Ty>),
}

impl Ast {
    fn codegen(&self) -> String {
        let mut s = "\
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>"
            .to_string();

        for item in &self.0 {
            if let Item::Struct { .. } = item {
                s.push_str(&format!("\n\n{}", item.codegen_forward_declaration()));
            }
        }

        for item in &self.0 {
            if let Item::Struct { .. } = item {
                s.push_str(&format!("\n\n{}", item.codegen()));
            }
        }

        for item in &self.0 {
            if let Item::Function(_) = item {
                s.push_str(&format!("\n\n{}", item.codegen_forward_declaration()));
            }
        }

        for item in &self.0 {
            if let Item::Function(f) = item {
                s.push_str(&format!("\n\n{}", f.codegen()));
            }
        }

        s
    }
}

impl Item {
    fn codegen(&self) -> String {
        match self {
            Item::Function(f) => f.codegen(),

            Item::Struct { name, fields } => {
                let mut s = format!("struct {name} {{");

                for (name, ty) in fields {
                    s.push_str(&format!("\n\t{} {name};", ty.codegen()));
                }

                s.push_str(&format!("\n}};"));

                s
            }
        }
    }

    fn codegen_forward_declaration(&self) -> String {
        match self {
            Item::Function(f) => format!("{};", f.codegen_signature()),
            Item::Struct { name, .. } => format!("typedef struct {name} {name};"),
        }
    }
}

impl Function {
    fn codegen(&self) -> String {
        let mut s = self.codegen_signature();

        s.push_str(" {\n");

        for statement in &self.body {
            s.push_str(&format!(
                "\t{}\n",
                statement.codegen().replace('\n', "\n\t")
            ));
        }

        s.push('}');

        s
    }

    fn codegen_signature(&self) -> String {
        let mut s = match &self.return_ty {
            Some(return_ty) => return_ty.codegen(),
            None => "void".to_string(),
        };

        s.push_str(&format!(" {}(", self.name));

        for (i, (name, ty)) in self.params.iter().enumerate() {
            if i != 0 {
                s.push_str(", ");
            }

            s.push_str(&format!("{} {name}", ty.codegen()));
        }

        s.push_str(")");

        s
    }
}

impl Statement {
    fn codegen(&self) -> String {
        match self {
            Statement::Let { name, ty, val } => {
                let mut s = format!("{} {name}", ty.codegen());

                if let Some(val) = val {
                    s.push_str(&format!(" = {}", val.codegen()));
                }

                s.push(';');

                s
            }
            Statement::Return { val } => {
                let mut s = "return".to_string();
                if let Some(val) = val {
                    s.push_str(&format!(" {}", val.codegen()));
                }
                s.push(';');
                s
            }
            Statement::Break => "break;".to_string(),
            Statement::If {
                condition,
                true_branch,
                false_branch,
            } => {
                let mut s = format!("if ({}) {{", condition.codegen());

                for statement in true_branch {
                    s.push_str(&format!(
                        "\n\t{}",
                        statement.codegen().replace('\n', "\n\t")
                    ));
                }

                s.push_str("\n}");

                if let Some(false_branch) = false_branch {
                    s.push_str(" else {");
                    for statement in false_branch {
                        s.push_str(&format!(
                            "\n\t{}",
                            statement.codegen().replace('\n', "\n\t")
                        ));
                    }
                    s.push_str("\n}");
                }

                s
            }
            Statement::Loop { body } => {
                let mut s = "while (true) {".to_string();

                for statement in body {
                    s.push_str(&format!(
                        "\n\t{}",
                        statement.codegen().replace('\n', "\n\t")
                    ));
                }

                s.push_str("\n}");

                s
            }
            Statement::Expr(e) => format!("{};", e.codegen()),
        }
    }
}

impl Expr {
    fn codegen(&self) -> String {
        match self {
            Expr::Variable(n) => n.clone(),
            Expr::Int(n) => n.to_string(),
            Expr::Char(c) => format!("'{c}'"),
            Expr::String(s) => format!("\"{s}\""),
            Expr::Unary { op, expr } => {
                let mut s = match op {
                    UnaryOp::Not => "!".to_string(),
                    UnaryOp::TakeAddress => "&".to_string(),
                    UnaryOp::Dereference => "*".to_string(),
                };

                s.push_str(&expr.codegen());

                s
            }
            Expr::Binary { lhs, op, rhs } => {
                let mut s = format!("({}", lhs.codegen());

                s.push(' ');
                match op {
                    BinaryOp::Add => s.push('+'),
                    BinaryOp::Sub => s.push('-'),
                    BinaryOp::Mul => s.push('*'),
                    BinaryOp::Div => s.push('/'),
                    BinaryOp::Assign => s.push('='),
                    BinaryOp::AddAssign => s.push_str("+="),
                    BinaryOp::SubAssign => s.push_str("-="),
                    BinaryOp::MulAssign => s.push_str("*="),
                    BinaryOp::DivAssign => s.push_str("/="),
                    BinaryOp::Eq => s.push_str("=="),
                    BinaryOp::NEq => s.push_str("!="),
                    BinaryOp::Lt => s.push('<'),
                    BinaryOp::Gt => s.push('>'),
                    BinaryOp::LtEq => s.push_str("<="),
                    BinaryOp::GtEq => s.push_str(">="),
                    BinaryOp::And => s.push_str("&&"),
                    BinaryOp::Or => s.push_str("||"),
                }
                s.push(' ');

                s.push_str(&format!("{})", rhs.codegen()));

                s
            }
            Expr::Call { function, args } => {
                let mut s = format!("{function}(");

                for (i, arg) in args.iter().enumerate() {
                    if i != 0 {
                        s.push_str(", ");
                    }

                    s.push_str(&arg.codegen());
                }

                s.push(')');

                s
            }
            Expr::FieldAccess {
                expr,
                field,
                through_pointer,
            } => {
                format!(
                    "{}{}{field}",
                    expr.codegen(),
                    if *through_pointer { "->" } else { "." }
                )
            }
        }
    }
}

impl Ty {
    fn codegen(&self) -> String {
        match self {
            Ty::Char => "char".to_string(),
            Ty::Int => "int".to_string(),
            Ty::Named(n) => n.clone(),
            Ty::Pointer(ty) => format!("{}*", ty.codegen()),
        }
    }
}
