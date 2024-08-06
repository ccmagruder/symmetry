#!/bin/zsh

earthly account login --token $(cat /run/secrets/user_earthly_token)
earthly org select ccmagruder
earthly sat select earthly-noble-arm64

earthly -i +test && earthly -a +build/compile_commands.json

