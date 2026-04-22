import React from 'react'
import { Box, Button, HStack, Text } from '@chakra-ui/react'

/**
 * DemoBox — Shows Demo 1 / Demo 2 buttons in a styled container.
 * Props:
 *   onDemo(n)  — called with 1 or 2 when a button is clicked
 *   loading    — disables buttons while running
 */
export default function DemoBox({ onDemo, loading }) {
    return (
        <Box
            bg="whiteAlpha.50" borderRadius="lg" p={3} mb={5}
            border="1px solid" borderColor="whiteAlpha.100"
        >
            <HStack spacing={3} align="center">
                <Text fontSize="xs" fontWeight="600" color="gray.400" mr={1}>
                    🎬 Demo nhanh
                </Text>
                <Button
                    size="sm" fontSize="xs" variant="outline" colorScheme="brand"
                    onClick={() => onDemo(1)} isDisabled={loading}
                >
                    Demo 1
                </Button>
                <Button
                    size="sm" fontSize="xs" variant="outline" colorScheme="brand"
                    onClick={() => onDemo(2)} isDisabled={loading}
                >
                    Demo 2
                </Button>
                <Text fontSize="xs" color="gray.500" ml={1}>
                    — Tự động load ảnh mẫu & chạy inference
                </Text>
            </HStack>
        </Box>
    )
}
