# Documentation Notes

We will use a master file (./source/index.rst) for all documentations.

## Dependencies
The following packages must be installed using either pip/conda within your activated virtual/conda environments. It is necessary to automatically and correctly generate the documentation and render it using the Read the Docs theme. 
  * sphinx
  * sphinx_rtd_theme
  * sphinx-autodoc-typehints


## Creating new documentation pages
To create additional documentation pages, create a new file in the `./source/` directory with any appropiate name `<doc_name>.rst`. For example, consider the documentation directory for `policy.rst`:

```
Policies
--------
In order to update each agent's action in the environment given its, we have to define a policy. Here are the policies we have implemented so far:

*   Boltzmann
*   E-greedy
*   Gaussian Exploration
*   ...
```

Sphinx needs to know about it, so in `index.rst`, edit the `.. toctree::` section to add the `policy` page:

```
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   policy
```

## Generate and render documentation

Save both pages and render the updated documentation by running `make <builder>`, where `<builder>` is one of the supported builders (i.e. html, latex, or linkcheck). For this project, we will be using html, so the command should be `make html`. 

The rendered pages will be located `docs/build/html/`. Opening `index.html` in your browser should show the current version of the documentation.

### Updated instructions
  - `cd docs`
  - `./make_docs.sh` 