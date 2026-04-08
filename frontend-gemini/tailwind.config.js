/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        obsidian: '#0b0b11',
        graphite: '#161922',
        ember: '#ff6a2a',
        auric: '#f2c96c',
        neon: '#3ef0d6',
      },
      fontFamily: {
        sans: ['"Space Grotesk"', 'sans-serif'],
        display: ['Unbounded', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
