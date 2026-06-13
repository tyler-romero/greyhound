.PHONY : docs
docs :
	uv run python scripts/generate_api_docs.py
	npm --prefix docs run dev

.PHONY : docs-install
docs-install :
	npm ci --prefix docs

.PHONY : docs-api
docs-api :
	uv run python scripts/generate_api_docs.py

.PHONY : docs-build
docs-build :
	uv run python scripts/generate_api_docs.py
	npm --prefix docs run check
	npm --prefix docs run build

.PHONY : checks
checks :
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check .

.PHONY : lint
lint :
	uv run ruff check .

.PHONY : typecheck
typecheck :
	uv run ty check .

.PHONY : style
style :
	uv run ruff format .

.PHONY : style-check
style-check :
	uv run ruff format --check .

.PHONY : test
test :
	uv run pytest -v --color=yes --doctest-modules src/tests/ src/greyhound/

.PHONY : build
build :
	rm -rf *.egg-info/ src/*.egg-info/ build/ dist/
	uv build

RELEASE_NOTES ?= RELEASE_NOTES.md

.PHONY : release-notes
release-notes :
	uv run python scripts/release_notes.py > $(RELEASE_NOTES)

.PHONY : publish
publish :
	uv publish

BASE_REF ?= origin/main

.PHONY : changelog-check
changelog-check :
	git diff --name-only $$(git merge-base $(BASE_REF) HEAD) | grep '^CHANGELOG.md$$' && echo "Thanks for helping keep our CHANGELOG up-to-date!"
