#!/bin/sh

if [ -f Makefile ]; then
    rm -f config.status # Yes, it's gross.
    touch config.status # Yes, it's even grosser.
    make distclean-generic
    rm -f config.status
    rm -f Makefile
fi

aclocal -I m4 &&
autoheader &&
automake --add-missing &&
autoconf
