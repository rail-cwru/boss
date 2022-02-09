# Python

We will be following [PEP8](https://www.python.org/dev/peps/pep-0008/) and [PEP257](https://www.python.org/dev/peps/pep-0257/) for styles and docstrings, respectively. The rest of the document outlines clarifications to the above conventions. 

Before commtting and pushing any code to the server, you should ensure you have followed the conventions described. You can use `make lint` from the top level directory to run a PEP8 check over all the source code. Alternatively, `make linc` can be used to check over ony the files you have changed.

## Indentation
Use 4 space, or set tabs equal to 4 spaces in your IDE of choice, for indentation. Not doing so will likely cause bugs.


## Docstrings
All non-trivial methods should have docstrings. As mentioned before, docstrings should follow [PEP257](https://www.python.org/dev/peps/pep-0257/) guidelines. We will primarily be using the [Numpy style](https://numpydoc.readthedocs.io/en/latest/format.html) documentation format. For more examples, view [this](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) website.

To summarize: There are two types of docstrings, short-form and long-form. 


### Short Form
A short-form docstring fits entirely on one line, including the triple quotes. It is used for simple functions, especially (though by no means exclusively) ones that are not part of a public API.
> """This is an examples of a one line doc-string."""

### Long Form
If the description spills past one line, you should move to the long-form docstring: a summary line (one physical line) starting with a triple-quote ("""), terminated by a period, question mark, or exclamation point, followed by a blank line, followed by the rest of the doc string starting at the same cursor position as the first quote of the first line, ending with triple-quotes on a line by themselves. (Unlike what the BDFL suggests in PEP 257, we do not put a blank line before the ending quotes.)

```python
"""This comment serves to demonstrate the format of a docstring.

Note that the summary line is always at most one line long, and
on the same line as the opening triple-quote, and with no spaces
between the triple-quote and the text.  Always use double-quotes
(") rather than single-quotes (') for your docstring.   (This
lets us reserve ''' (3 single-quotes) as a way of commenting out
big blocks of code.
"""
```

The docstring should describe the function's calling syntax and its semantics, not its implementation. Additionally, it should end with the following special sections (see [examples](https://numpydoc.readthedocs.io/en/latest/format.html) from the Numpy style guide for more details).

  * **Params**: List each parameter by name, and a description of it. The description can span several lines (use a hanging indent if so). Use instead of "Args".
  * **Returns** (or **Yields** for generators): Describe the type and semantics of the return value. If the function only returns None, this section is not required.
  * **Raises**: List all exceptions that are relevant to the interface. If the function does not have any exceptions, this section is not required.


## Symbol (Class, Function, Variable) Naming
When naming a top-level symbol -- function, class, or variable -- use a leading underscore if the symbol is private to the module: that is, nobody may reference that symbol except for code in the module itself and its associate test file(s).

Modules themselves may have names starting with a leading underscore to mean that all symbols in that module are private to the package the symbols is in. A "package" is basically the top-level directory that the module is under. (Example packages are "agentsystem", "callbacks", and "algorithm") Note that an underscored symbol within an underscored module is still considered private to the module.

Inside a class, use a leading underscore to indicate a symbol (method or class variable) is private to that class and its subclasses. If you want a symbol to be private to the class itself, and not even visible to subclasses, use a leading double-underscore.