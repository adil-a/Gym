#!/bin/bash
# Append build-artifact ignore patterns for the given language to ./.gitignore, so the agent's
# final patch (captured by run_openclaw.sh) doesn't carry compiled/generated junk. Mirrors the
# OpenHands fork's per-language gitignore step, unified into one language-parametrized script.
#
# Usage: gitignore.sh <language>
#   - known build-artifact language  -> append its patterns, exit 0
#   - unknown language (python/ruby/php/...) -> skip: print notice, exit 1
# run_openclaw.sh keys its `_ng_gitignore_ran` flag (and thus the `.gitignore` diff-exclusion)
# on this exit code, so an unsupported language never causes a legit .gitignore edit to be dropped.

lang="$1"

declare -a ignores
case "$lang" in
    c | cpp)
        ignores=(
            "build/" "Build/" "bin/" "lib/"
            "*.o" "*.out" "*.obj" "*.exe" "*.dll" "*.so" "*.dylib"
        )
        ;;
    go)
        ignores=("pkg/" "vendor/" "bin/" "*.exe" "*.test")
        ;;
    java)
        ignores=("target/" "out/" "*.class" "*.jar" ".gradle/")
        ;;
    rust)
        ignores=("target/" "Cargo.lock" "**/*.rs.bk")
        ;;
    javascript)
        ignores=(
            "node_modules/" "build/" "dist/" ".next/" "coverage/" ".env"
            "npm-debug.log*" "yarn-debug.log*" "yarn-error.log*"
        )
        ;;
    typescript)
        ignores=(
            "node_modules/" "build/" "dist/" ".next/" "coverage/" ".env"
            "npm-debug.log*" "yarn-debug.log*" "yarn-error.log*"
            "*.js" "*.js.map" "*.d.ts" ".tsbuildinfo"
        )
        ;;
    *)
        echo "No gitignore patterns for language '$lang'"
        exit 1
        ;;
esac

if [ ! -f .gitignore ]; then
    touch .gitignore
    echo "Created new .gitignore file"
else
    echo >> .gitignore
fi

added=0
existing=0
for ignore in "${ignores[@]}"; do
    if ! grep -Fxq "$ignore" .gitignore; then
        echo "$ignore" >> .gitignore
        ((added++))
    else
        ((existing++))
    fi
done

echo "Added $added new entries to .gitignore"
echo "Found $existing existing entries"
echo "Done! .gitignore has been updated."
