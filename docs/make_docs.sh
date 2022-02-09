#!/bin/bash

# NOTE: This is very hacky workaround
# TODO: Change rootdir to relative import based off where this script is being called from. Currently, this is hacked together.
# TODO: Potentially make it so that the modules are 
# TODO: Scrap this whole process and figure out how to use automodule properly (see... https://romanvm.pythonanywhere.com/post/autodocumenting-your-python-code-sphinx-part-ii-6/). I couldn't get this to work for some reason. 

# Find all directories one level deep within the ROOTDIR, which is just the `src/` folder. 
# These are all the modules in the framework.
OUTDIR="./source"
SUBDIRS="`find ../src -type d -maxdepth 1 -mindepth 1`"
for file in $SUBDIRS; do
    # Make all the necessary docs, remove `--force` if you don't want to rewrite all the docs
    echo "Making .rst templates for $file"
    sphinx-apidoc --force -o $OUTDIR $file
done

# Remove modules.rst as it is unnecessary
rm ${OUTDIR}/modules.rst

# Make the docs
make html