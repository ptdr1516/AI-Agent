/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bgMain: '#0f1115',      // Very dark slate/blue
        bgSidebar: '#171921',   // Slightly lighter for sidebar
        bgPanel: '#1a1c23',     // For tools panel
        bgHover: '#252830',
        bgBorder: '#2b2e38',
        accentMain: '#3b82f6',  // Blue primary
        accentGlow: 'rgba(59, 130, 246, 0.15)',
        textMain: '#f3f4f6',
        textMuted: '#9ca3af',
        textFaint: '#6b7280',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
      animation: {
        'pulse-fast': 'pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out forwards',
        'slide-up': 'slideUp 0.3s ease-out forwards',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        }
      }
    },
  },
  plugins: [],
}
