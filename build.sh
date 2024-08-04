#!/bin/zsh

earthly sat ls

if $?
then
  earthly account login --token $(cat /run/secrets/user_earthly_token)
  earthly org select ccmagruder
  earthly sat select earthly-noble-arm64
fi

earthly +test

