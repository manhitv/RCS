TECTONIC = /home/s224852302/.conda/envs/latex/bin/tectonic
MAIN     = main.tex

.PHONY: all pdf clean distclean

all: pdf

pdf:
	$(TECTONIC) $(MAIN)

clean:
	rm -f main.aux main.bbl main.blg main.log main.out \
	      main.toc main.fls main.fdb_latexmk main.synctex.gz

distclean: clean
	rm -f main.pdf
