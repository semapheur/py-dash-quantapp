/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/app.py',
    './src/pages/**/*.py',
    './src/components/**/*.py',
    './src/assets/*.js',
  ],
  theme: {
    extend: {
      colors: {
        primary: 'rgb(var(--color-primary) / <alpha-value>)',
        secondary: 'rgb(var(--color-secondary) / <alpha-value>)',
        text: 'rgb(var(--color-text) / <alpha-value>)',
      },
    },
  },
  plugins: [],
}
