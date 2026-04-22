import React, { useState, useCallback, useEffect, useRef } from 'react'
import {
  Box, Container, Flex, Heading, HStack, Badge,
  Tabs, TabList, Tab, TabPanels, TabPanel, Tooltip,
} from '@chakra-ui/react'
import Task1Tab from './components/Task1Tab.jsx'
import Task2Tab from './components/Task2Tab.jsx'
import Task3Tab from './components/Task3Tab.jsx'

export default function App() {
  const [tabIndex, setTabIndex] = useState(0)
  const [health, setHealth] = useState({ active_task: null, gpu_available: false })

  useEffect(() => {
    const poll = () => {
      if (document.visibilityState === 'hidden') return
      fetch('/api/health')
        .then((r) => r.json())
        .then(setHealth)
        .catch(() => { })
    }
    poll()
    const id = setInterval(poll, 60000)
    return () => clearInterval(id)
  }, [])

  const handleTab = useCallback((i) => {
    setTabIndex(i)
    // Không gọi activate ở đây — mỗi tab tự activate khi bấm nút Xử lý
  }, [])

  const task2Ref = useRef(null)
  const task3Ref = useRef(null)

  const handleSendToTask2 = useCallback(async (imageUrl) => {
    await task2Ref.current?.loadExternalImage(imageUrl)
    setTabIndex(1)
  }, [])

  const handleSendToTask3 = useCallback(async (imageUrl) => {
    await task3Ref.current?.loadExternalImage(imageUrl)
    setTabIndex(2)
  }, [])

  return (
    <Container maxW="1500px" py={6}>
      {/* Header */}
      <Flex justify="space-between" align="center" mb={6}>
        <Heading size="lg" fontWeight="700">
          🎨 Image Processing Studio
        </Heading>
        <HStack spacing={3}>
          <Tooltip label={health.gpu_available ? 'GPU is available' : 'No GPU detected'}>
            <Badge
              colorScheme={health.gpu_available ? 'green' : 'red'}
              variant="subtle"
              px={3} py={1} borderRadius="full" fontSize="xs"
            >
              {health.gpu_available ? '● GPU' : '○ No GPU'}
            </Badge>
          </Tooltip>
          {health.active_task && (
            <Tooltip label="Currently active model on GPU">
              <Badge
                colorScheme="purple" variant="subtle"
                px={3} py={1} borderRadius="full" fontSize="xs"
              >
                Model: {health.active_task.toUpperCase()}
              </Badge>
            </Tooltip>
          )}
        </HStack>
      </Flex>

      {/* Tabs */}
      <Tabs
        index={tabIndex}
        onChange={handleTab}
        variant="soft-rounded"
        colorScheme="brand"
        isLazy
        lazyBehavior="keepMounted"
      >
        <TabList
          bg="whiteAlpha.50" p={1.5} borderRadius="xl" mb={6}
          overflowX="auto" flexWrap="nowrap"
        >
          <Tab fontSize="sm" whiteSpace="nowrap">🖼️ Multi-Exposure Fusion</Tab>
          <Tab fontSize="sm" whiteSpace="nowrap">🎨 Inpainting & Editing</Tab>
          <Tab fontSize="sm" whiteSpace="nowrap">🔲 Outpainting</Tab>
        </TabList>

        <TabPanels>
          <TabPanel p={0}><Task1Tab onSendToTask2={handleSendToTask2} onSendToTask3={handleSendToTask3} /></TabPanel>
          <TabPanel p={0}><Task2Tab ref={task2Ref} onSendToTask3={handleSendToTask3} /></TabPanel>
          <TabPanel p={0}><Task3Tab ref={task3Ref} /></TabPanel>
        </TabPanels>
      </Tabs>
    </Container>
  )
}
