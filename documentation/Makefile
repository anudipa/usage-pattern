START = xxxnote nodraft noblue
END = missing
CLASS = $(PYTEX)/cls/proposal1.cls

all: paper ABSTRACT wc

open: paper.pdf

wc: abstract.tex

ABSTRACT: $(PYTEX)/bin/clean $(PYTEX)/bin/lib.py abstract.tex
	@$(PYTEX)/bin/clean abstract.tex ABSTRACT

clean: rulesclean
	@rm -f ABSTRACT

include $(PYTEX)/make/Makerules

spellcheck: .spellcheck | silent

.spellcheck: $(PAPER_TEXTILES) .okwords
	@hunspell -t -l -p $(PWD)/.okwords $(PAPER_TEXFILES) | sort -f | uniq | tee badwords && touch .spellcheck


silent:
	@:

.PHONY:
	spellcheck
