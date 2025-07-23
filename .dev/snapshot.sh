#!/bin/bash

# Sprawdzenie, czy podano nazwę w parametrze
if [ -z "$1" ]; then
  echo "Użycie: $0 NAZWA_PATCHA"
  exit 1
fi

# Generowanie znacznika czasu
timestamp=$(date +"%Y%m%d_%H%M%S")

# Nazwa pliku patcha
patch_name="${timestamp}_$1.patch"

# Katalog docelowy
destination=".dev/timeline"

# Sprawdzenie, czy katalog istnieje
if [ ! -d "$destination" ]; then
  echo "Katalog $destination nie istnieje. Tworzenie..."
  mkdir -p "$destination"
fi

# Tworzenie patcha z niewkomitowanych zmian (również staged)
git diff > "$destination/$patch_name"

# Informacja zwrotna
echo "Utworzono patch: $destination/$patch_name"