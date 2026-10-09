# Makefile for the PICASO documentation.
#
#   make html            build with notebook execution (cached; needs full reference data)
#   make html-noexec     build without running notebooks (what CI does)
#   make html-force      rerun every notebook, ignoring the cache
#   make deploy          build, then publish to the root of gh-pages ("latest")
#   make deploy-release  same, plus a frozen copy at /v<version>/ for the version switcher
#   make clean-cache     delete the notebook execution cache
#
# See README.md in this folder for the full workflow.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html-noexec html-force deploy deploy-release clean-cache Makefile

html-noexec:
	PICASO_DOCS_EXECUTE=off $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-force:
	PICASO_DOCS_EXECUTE=force $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

deploy: html
	./deploy.sh

deploy-release: html
	./deploy.sh --release

clean-cache:
	rm -rf "$(BUILDDIR)/.jupyter_cache"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
