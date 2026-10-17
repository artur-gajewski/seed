# Repository Guidelines

## Project Structure & Module Organization

Seed is a local language-model training studio using React, TypeScript, Vite, and Node.js 24.

- `src/` contains the UI: `components/`, `hooks/`, `lib/api.ts`, styles, and sample text in `data/`.
- `server/` contains HTTP routing, training orchestration, checkpoint storage, and the Transformer engine. Keep routing separate from model and training logic.
- `shared/api.ts` defines shared API types.
- `tests/` covers models, tensor operations, chat, HTTP, and studio behavior.
- `corpus-examples/` contains importable training text.
- `dist/` is generated frontend output; `data/checkpoint.json` is runtime model storage.

## Build, Test, and Development Commands

Use `nvm use` to select Node 24, then `npm ci` to install dependencies.

- `npm run dev`: Start Vite on port 5173 and the development API on port 3001.
- `npm run build`: Run TypeScript checking and build the frontend.
- `npm start`: Serve the built frontend and API.
- `npm test`: Run all tests with Node's built-in test runner.
- `npm run typecheck`: Check TypeScript without emitting files.
- `npm run format:check`: Check Prettier formatting; `npm run format` rewrites files.
- `docker compose up --build -d`: Build and run the containerized studio.

## Coding Style & Naming Conventions

Use strict TypeScript and ES modules. Follow Prettier: two-space indentation, single quotes, semicolons, trailing commas, and a 100-character print width. Use PascalCase for React components and classes, camelCase for functions and variables, and `use` prefixes for hooks. Follow existing explicit `.ts` extensions in server imports. Keep API contracts in `shared/api.ts`.

## Testing Guidelines

Tests use `node:test` and `node:assert/strict`; name files `tests/*.test.ts`. No coverage threshold is configured. Add meaningful regression tests for behavior changes, especially gradients, training, checkpoint restoration, and request validation. Use small models and bounded training loops. Run tests, type checking, and formatting checks before submitting code changes.

## Commit & Pull Request Guidelines

History uses short, imperative subjects such as “Add button to delete checkpoint file”; no mandatory commit prefix is established. Keep commits focused. PRs should explain the problem, resulting behavior, and validation performed. Link relevant issues and include screenshots for visible UI changes.

## Configuration & Data

The API defaults to `127.0.0.1:3001`; `HOST` and `PORT` override binding. Preserve origin validation. Do not commit generated checkpoints, dependencies, or build output. Keep training corpora separately: checkpoints do not include their source text.
