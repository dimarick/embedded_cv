#!/bin/bash
# Включаем поддержку **
shopt -s globstar nullglob

FILES=`ls src/**/*.h | grep -Ev '.*calibrator.*' | grep -Ev '.*mini_server.*' | xargs -n 1 -I{} echo -i {}`
hpp2plantuml $FILES -d -o /tmp/classes.puml
cat /tmp/classes.puml | grep -Ev '^\s+-' | grep -Ev '_DIM_' | grep -Eo '^\s*.{0,80}' | sed -e '/^@startuml$/a\left to right direction' > classes_main.puml

FILES=`ls src/calibrator/**/*.h | xargs -n 1 -I{} echo -i {}`
hpp2plantuml $FILES -d -o /tmp/classes.puml
cat /tmp/classes.puml | grep -Ev '^\s+-' | grep -Ev '_DIM_' | grep -Eo '^\s*.{0,80}' > classes_calibrator.puml

FILES=`ls src/mini_server/**/*.h | xargs -n 1 -I{} echo -i {}`
hpp2plantuml $FILES -d -o /tmp/classes.puml
cat /tmp/classes.puml | grep -Ev '^\s+-' | grep -Eo '(^\s+.{0,80})|(\S.*)' | sed -e '/^@startuml$/a\left to right direction' > classes_mini_server.puml

FILES=`ls ws_ctl/**/*.h | grep -Ev '.*build.*' | grep -Ev '.*web.*' | grep -Ev '.*3rdparty.*' | grep -Ev '.*release.*' | xargs -n 1 -I{} echo -i {}`
hpp2plantuml $FILES -d -o /tmp/classes.puml
cat /tmp/classes.puml | grep -Ev '^\s+-' > classes_ws_ctl.puml

plantuml -tsvg classes_*.puml
