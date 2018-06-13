# Makefile for papers
SHELL:=/bin/bash
BIB = src/mvf
PAPER = mvf



# figures that are committed into the repository directly:
MANUAL_FIGURES = $(shell find . -name  '*.png')


.DEFAULT_GOAL := all
all: $(PAPER).pdf


# skip dvi and ps, as some figures are only availabe in pdf
$(PAPER).pdf: $(PAPER).tex  $(MANUAL_FIGURES) $(BIB).bib
	pdflatex -draftmode -shell-escape -file-line-error -halt-on-error $<
	bibtex $(PAPER)
	pdflatex -draftmode -shell-escape -file-line-error -halt-on-error $<
	pdflatex -shell-escape -file-line-error -halt-on-error $<
	make clean $<




# The || : is there to make sure that subsequent lines get executed even if
# some of the removes fail because the files to be removed do not exist.
clean:
	@rm -f *.aux || :
	@rm -f *.dvi || :
	@rm -f *.out || :
	@rm -f *.log || :
	@rm -f *.bbl || :
	@rm -f *.blg || :
	@rm -f *.backup || :
	@rm -f *~ || :
	@rm -f *.fls || :
	@rm -f *synctex.gz || :
	@rm -f *.fdb_latexmk || :
	@rm -f *Notes.bib || :
