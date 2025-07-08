# Constraint Lattice Dashboard

A modern, high-performance dashboard built with React, TypeScript, and Vite, featuring advanced visualization capabilities using D3.js and Framer Motion.

## Features

- Blazing fast development with Vite
- Beautiful, responsive UI with modern design
- Advanced data visualization with D3.js
- Smooth animations with Framer Motion
- Type-safe code with TypeScript
- Comprehensive testing setup
- Code quality with ESLint and Prettier
- Optimized production builds

## Development

### Prerequisites

- Node.js 18+ & npm 8+
- Git

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/constraint-lattice.git
   cd constraint-lattice
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   
   The app will be available at `http://localhost:3001`

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier
- `npm run type-check` - Check TypeScript types
- `npm run analyze` - Analyze bundle size

## Project Structure

The Constraint Lattice Dashboard follows a modular and well-organized structure to ensure maintainability and scalability. Below is a detailed breakdown of the project's directory structure and key files:

### Top-Level Directories

- `public/`: Contains static assets that are publicly available, such as images, fonts, and the `index.html` file which serves as the entry point.
- `src/`: The main source code directory, housing all application logic, components, and styles.

### Source Code (`src/`) Breakdown

#### `components/`
This directory contains reusable UI components that are used across multiple views. Components are typically organized by feature or domain. Examples include:
- `ui/`: Foundational UI elements (buttons, cards, tabs, etc.)
- `charts/`: Data visualization components built with D3.js
- `layout/`: Layout components (headers, sidebars, etc.)

#### `contexts/`
React context providers for global state management. Each context typically corresponds to a specific domain of the application state, such as:
- `AuthContext.tsx`: Manages user authentication state
- `ConstraintContext.tsx`: Provides access to the constraint lattice and operations

#### `hooks/`
Custom React hooks for encapsulating reusable logic. Examples:
- `useConstraintResolver.ts`: Hook for constraint resolution logic
- `useDataFetching.ts`: Hook for data fetching and caching

#### `services/`
Services for interacting with APIs and external systems. Key files:
- `centralControllerService.ts`: Core service for constraint orchestration and management
- `geminiService.ts`: Service for interacting with the Gemini AI API for constraint analysis

#### `views/`
Top-level page components that represent different perspectives of the constraint system. Each view is a self-contained feature module:
- `OrchestrationView.tsx`: View for constraint orchestration and workflow management
- `PhenomenologicalView.tsx`: View for analyzing emergent system behaviors
- `SecurityView.tsx`: View for security constraint management
- `StructuralView.tsx`: View for visualizing constraint relationships
- `SymbolicView.tsx`: View for symbolic constraint manipulation
- `VarkielAgentView.tsx`: View for interacting with autonomous constraint-solving agents

#### `utils/`
Utility functions and helpers used throughout the application. Includes:
- `constraintUtils.ts`: Functions for constraint manipulation and analysis
- `apiClient.ts`: Axios instance for API requests
- `animationUtils.ts`: Helper functions for Framer Motion animations

### Configuration Files

- `vite.config.ts`: Configuration for the Vite build system
- `tsconfig.json`: TypeScript configuration
- `netlify.toml`: Deployment configuration for Netlify
- `.eslintrc.js`: ESLint configuration for code quality
- `.prettierrc`: Prettier configuration for code formatting

### Other Important Files

- `package.json`: Defines project dependencies and scripts
- `README.md`: Project documentation (this file)
- `windsurf_deployment.yaml`: Configuration for Windsurf deployment (if applicable)

## Configuration

### Environment Variables

Create a `.env.local` file in the root directory:

```env
VITE_API_URL=https://api.example.com
VITE_APP_ENV=development
```

### Build Configuration

- Vite configuration: `vite.config.ts`
- TypeScript configuration: `tsconfig.json`
- Netlify configuration: `netlify.toml`

## Deployment

The project is configured for deployment on Netlify. To deploy:

1. Push your code to a Git repository
2. Connect the repository to Netlify
3. The build settings are pre-configured in `netlify.toml`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Vite](https://vitejs.dev/) - Next Generation Frontend Tooling
- [React](https://reactjs.org/) - A JavaScript library for building user interfaces
- [TypeScript](https://www.typescriptlang.org/) - Typed JavaScript at Any Scale
- [D3.js](https://d3js.org/) - Data-Driven Documents
- [Framer Motion](https://www.framer.com/motion/) - Production-ready animation library for React
- [Netlify](https://www.netlify.com/) - The fastest way to combine your favorite tools and APIs to build the fastest sites, stores, and apps for the web.
