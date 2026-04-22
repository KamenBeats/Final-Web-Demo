import { extendTheme } from '@chakra-ui/react'

const theme = extendTheme({
  config: {
    initialColorMode: 'dark',
    useSystemColorMode: false,
  },
  fonts: {
    heading: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    body: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  colors: {
    brand: {
      50: '#eef2ff',
      100: '#e0e7ff',
      200: '#c7d2fe',
      300: '#a5b4fc',
      400: '#818cf8',
      500: '#6366f1',
      600: '#4f46e5',
      700: '#4338ca',
      800: '#3730a3',
      900: '#312e81',
    },
  },
  styles: {
    global: {
      body: {
        bg: '#0b0f19',
        color: '#e2e8f0',
      },
    },
  },
  components: {
    Tabs: {
      variants: {
        'soft-rounded': {
          tab: {
            fontWeight: '500',
            color: 'gray.400',
            _selected: {
              bg: 'brand.500',
              color: 'white',
              fontWeight: '600',
            },
          },
        },
      },
    },
  },
})

export default theme
