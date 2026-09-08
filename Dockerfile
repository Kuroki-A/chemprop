FROM mambaorg/micromamba:2.8.1

# environment.yml ends with an editable installation of this checkout, so the
# source tree must exist before micromamba invokes pip. The previous Dockerfile
# copied only environment.yml at this point and therefore attempted ``-e .``
# from a directory that was not a Python project.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/chemprop
WORKDIR /opt/chemprop

RUN micromamba install --yes --name base --file environment.yml && \
    micromamba clean --all --yes
