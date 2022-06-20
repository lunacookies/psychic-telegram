# psychic-telegram

## Todo

- [x] C transpiler written in Rust
- [x] Rewrite transpiler in psychic-telegram
- [ ] Replace transpiler with proper compiler that emits C
  - [x] Lexer
  - [x] Syntax tree library
  - [ ] Event-based parser
    - Need special syntax for declaring C functions,
      which for now won’t have checked types.
      This allows for variadic functions and functions that take `void*`
      to be used without any implementation effort.
  - [ ] Name resolution
  - [ ] Type checking
- [ ] Add language features to make compiler development less painful
  - [ ] Enums
  - [ ] Methods
  - [ ] Dynamically-sized slices
  - [ ] Length-prefixed strings
- [ ] Improve compiler
  - [ ] Error tolerant parser
  - [ ] Pretty error messages
- [ ] Add more language features
  - [ ] `when`
  - [ ] Union types
  - [ ] Struct literals
  - [ ] Forbid uninitialised variables
  - [ ] for-each loops
