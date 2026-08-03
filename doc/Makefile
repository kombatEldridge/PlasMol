# Minimal makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = _build
# conf.py lives in doc/ (parent of SOURCEDIR)
CONFDIR       = .

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS)

.PHONY: help Makefile html clean livehtml

html:
	@$(SPHINXBUILD) -b html -c "$(CONFDIR)" "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

# Optional: pip install sphinx-autobuild
livehtml:
	sphinx-autobuild -c "$(CONFDIR)" "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Catch-all
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS)
