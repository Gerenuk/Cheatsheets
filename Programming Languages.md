Popular
=======
* Rust
* Go
* Elixir
* Scala
* (Clojure)
* Kotlin (Android)

Erlang
======
* functions in sandboxes
* hot-plugging
* no dropped data
* millions of processes
* Prolog-like syntax obsolete?

Elixir
======
* runs on Erlang VM
* functional
* hot for concurrent programming
* shared-nothing actor based
* metaprogramming by macros
* polymorphism by protocols
* pipeline operator, pattern matching, streams, mix
* web framework Phoenix
* immutable data
* efficient resource management
* hot-swappable code

Go
==
* like C, but easier on types and malloc
* kept simple (to be learnable)
* fast compiling
* efficient string library (-> analyze websites)
* good concurrency, e.g. multicore
* garbage collection
* static types but no hierarchy
* Communicating Sequential Processes for concurrency

Groovy
======
* dynamic language on JVM
* closures, overloading, polymorphic iteration, easy null pointer check

OCaml
=====
* complex type/data hierarchies

CoffeeScript
============
* Javascript transpiler
* cleaner syntax, less symbols

Scala
=====
* functional programming on JVM

Coconut
=======
* superset of Python
* compiles to Python
* introduces functional notation

Dart
====
* Google's new web language
* good syntax
* don't need JQuery
* data types
* syntactic shorthands
* had to compile to JS as well

Haskell
=======
* pure functional
* some integrated (Python: Scotch)
* for complex data structures

Julia
=====
* fast algorithms
* type inference
* metaprogramming for extension
* easy parallel distribution
* run-time tags, not compile-time types
* performance and expressiveness, not safety or correctness
* generic function is primary unit of abstraction; dynamic dispatch(?)
* immutable values

Rust
====
* zero-cost concurrency checks (compile- time)
* no GC; manual memory management
* every data has owner
* safe due to compile-time checks

Go vs Scala
===========
https://www.quora.com/Scala-vs-Go-Could-people-help-compare-contrast-these-on-relative-merits-demerits
* Go:
  * small set of orthogonal primitives
  * only CSP concurrency; easy to reason, but not that great
  * better IDE?
* Scala:
  * powerful but too complex for average programmer
  * more sophisticated concurrency
  * sometimes slow compilation (but see alternatives like Dotty)

Oz
==
* concurreny oriented
* contraint programming, distributed programming

P language
==========
* state machines that communicate

Parasail
========
* concurrency

Red
===
* for system programming

Io
==
* actor based concurrency

Elm
===
* web programming
* transpiles to JS
* most errors statically found
* very helpful compiler
* semantic versioning automatically
* fast, functional

Futhark
=======


Concurrency
===========
* Go: Promising
* Node.js: wide-spread
* Elixir: small, innovative; single threas BEAM may be slow
* Python: Twisted, greenlet, gevent, Stackless
* Scala: Akka
* Other:
  * join calculus: JoCaml, Join Java, JErland, C++ via Boost
