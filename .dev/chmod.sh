#!/bin/bash

# Nadaje chmod +x wszystkim plikom (nie katalogom) w bieżącym katalogu
for file in *; do
  if [ -f "$file" ]; then
    chmod +x "$file"
    echo "Dodano +x dla: $file"
  fi
done
