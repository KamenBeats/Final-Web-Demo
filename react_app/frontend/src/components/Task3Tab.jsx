import React, { useState, useRef, useCallback, useEffect, forwardRef, useImperativeHandle } from 'react'
import {
  Box, Button, SimpleGrid, Text, VStack, HStack, Center, Flex,
  Spinner, Textarea, useToast, Select,
  Slider, SliderTrack, SliderFilledTrack, SliderThumb,
  Accordion, AccordionItem, AccordionButton, AccordionPanel, AccordionIcon,
  Tabs, TabList, Tab,
  NumberInput, NumberInputField, NumberInputStepper,
  NumberIncrementStepper, NumberDecrementStepper,
} from '@chakra-ui/react'
import { ZoomableImage } from './ImageLightbox.jsx'
import DemoBox from './DemoBox.jsx'

const SDXL_BUCKETS = ['1:1', '4:3', '3:4', '16:9', '9:16', 'Customize']
const RESIZE_OPTS = ['Full', '50%', '33%', '25%']
const ALIGN_OPTS = ['Middle', 'Left', 'Right', 'Top', 'Bottom']
const BOX_H = 500

/* Which custom-pad directions are locked by position */
const getLockedDirs = (alignment) => {
  switch (alignment) {
    case 'Top': return { top: true }
    case 'Bottom': return { bottom: true }
    case 'Left': return { left: true }
    case 'Right': return { right: true }
    default: return {}
  }
}

const Task3Tab = forwardRef(function Task3Tab(props, ref) {
  const [imageFile, setImageFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [info, setInfo] = useState('')
  const [loading, setLoading] = useState(false)
  const [inputTab, setInputTab] = useState(0)

  const [targetRes, setTargetRes] = useState('1:1')
  const [alignment, setAlignment] = useState('Middle')
  const [resizeOption, setResizeOption] = useState('Full')
  const [prompt, setPrompt] = useState('')
  const [numSteps, setNumSteps] = useState(20)
  const [sharpen, setSharpen] = useState(1.0)
  const [loraScale, setLoraScale] = useState(0.0)

  const [overlapTop, setOverlapTop] = useState(10)
  const [overlapBottom, setOverlapBottom] = useState(10)
  const [overlapLeft, setOverlapLeft] = useState(10)
  const [overlapRight, setOverlapRight] = useState(10)

  const [padTop, setPadTop] = useState(0)
  const [padBottom, setPadBottom] = useState(0)
  const [padLeft, setPadLeft] = useState(0)
  const [padRight, setPadRight] = useState(0)

  const [history, setHistory] = useState([])
  const fileRef = useRef(null)
  const toast = useToast()

  /* Reset locked pad directions when alignment changes */
  useEffect(() => {
    const locked = getLockedDirs(alignment)
    if (locked.top) setPadTop(0)
    if (locked.bottom) setPadBottom(0)
    if (locked.left) setPadLeft(0)
    if (locked.right) setPadRight(0)
  }, [alignment])

  const loadExternalImage = useCallback(async (blobUrl) => {
    try {
      const resp = await fetch(blobUrl)
      const blob = await resp.blob()
      const file = new File([blob], 'image.png', { type: blob.type })
      setImageFile(file)
      setImageUrl(URL.createObjectURL(blob))
      setResultUrl(null)
      setPreviewUrl(null)
      setInfo('')
    } catch { /* ignore */ }
  }, [])

  useImperativeHandle(ref, () => ({ loadExternalImage }), [loadExternalImage])

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      setImageUrl(URL.createObjectURL(file))
      setResultUrl(null)
      setPreviewUrl(null)
      setInfo('')
      setInputTab(0)
    }
  }, [])

  /* Drag-and-drop */
  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const file = e.dataTransfer?.files?.[0]
    if (file && file.type.startsWith('image/')) {
      setImageFile(file)
      setImageUrl(URL.createObjectURL(file))
      setResultUrl(null)
      setPreviewUrl(null)
      setInfo('')
      setInputTab(0)
    }
  }, [])
  const handleDragOver = useCallback((e) => e.preventDefault(), [])

  const handleClearImage = useCallback(() => {
    setImageFile(null)
    setImageUrl(null)
    setPreviewUrl(null)
    setResultUrl(null)
    setInfo('')
  }, [])

  const buildFormData = useCallback(() => {
    const fd = new FormData()
    fd.append('image', imageFile)
    fd.append('target_res', targetRes)
    fd.append('custom_w', 1024)
    fd.append('custom_h', 1024)
    fd.append('alignment', alignment)
    fd.append('resize_option', resizeOption)
    fd.append('custom_resize_pct', 100)
    fd.append('overlap_percentage', 10)
    fd.append('overlap_left', overlapLeft)
    fd.append('overlap_right', overlapRight)
    fd.append('overlap_top', overlapTop)
    fd.append('overlap_bottom', overlapBottom)
    fd.append('pad_left', padLeft)
    fd.append('pad_right', padRight)
    fd.append('pad_top', padTop)
    fd.append('pad_bottom', padBottom)
    return fd
  }, [
    imageFile, targetRes, alignment, resizeOption,
    overlapLeft, overlapRight, overlapTop, overlapBottom,
    padLeft, padRight, padTop, padBottom,
  ])

  /* Fetch preview only when Preview tab is active, with debounce */
  useEffect(() => {
    if (!imageFile || inputTab !== 1) return
    const timer = setTimeout(async () => {
      try {
        const fd = buildFormData()
        const resp = await fetch('/api/task3/preview', { method: 'POST', body: fd })
        if (resp.ok) setPreviewUrl(URL.createObjectURL(await resp.blob()))
      } catch {
        // ignore preview errors
      }
    }, 500)
    return () => clearTimeout(timer)
  }, [buildFormData, imageFile, inputTab])

  const handleRun = async () => {
    if (!imageFile) return
    setLoading(true)
    setResultUrl(null)
    setInfo('')
    try {
      await fetch('/api/activate/task3', { method: 'POST' }).catch(() => { })
      const fd = buildFormData()
      fd.append('prompt', prompt)
      fd.append('num_steps', numSteps)
      fd.append('sharpen', sharpen)
      fd.append('lora_scale', loraScale)
      const t0 = Date.now()

      const submitResp = await fetch('/api/task3/run', { method: 'POST', body: fd })
      if (!submitResp.ok) throw new Error(await submitResp.text())
      const { job_id } = await submitResp.json()

      while (true) {
        await new Promise(r => setTimeout(r, 10000))
        const statusResp = await fetch(`/api/job/${job_id}/status`)
        if (!statusResp.ok) throw new Error('Lost job status')
        const { status, error } = await statusResp.json()
        if (status === 'error') throw new Error(error || 'Inference failed')
        if (status === 'done') break
      }

      const resultResp = await fetch(`/api/job/${job_id}/result`)
      if (!resultResp.ok) throw new Error(await resultResp.text())
      const url = URL.createObjectURL(await resultResp.blob())
      setResultUrl(url)
      setInfo(`Hoàn thành trong ${((Date.now() - t0) / 1000).toFixed(1)}s · ${numSteps} steps`)
      setHistory((prev) => [{ url }, ...prev.slice(0, 19)])
    } catch (err) {
      toast({ title: 'Error', description: err.message, status: 'error', duration: 5000 })
    } finally {
      setLoading(false)
    }
  }

  const DEMO_CONFIGS = {
    1: { image: '1/1.jpg', prompt: '1/1.txt' },
    2: { image: '2/2.jpg', prompt: '2/2.txt' },
  }

  const handleDemo = async (n) => {
    const demoCfg = DEMO_CONFIGS[n]
    if (!demoCfg) return
    try {
      // Fetch image
      const imgResp = await fetch(`/api/examples/task3/${demoCfg.image}`)
      if (!imgResp.ok) throw new Error('Failed to load demo image')
      const imgBlob = await imgResp.blob()
      const file = new File([imgBlob], demoCfg.image.split('/').pop(), { type: imgBlob.type })
      const url = URL.createObjectURL(imgBlob)

      // Fetch prompt
      const promptResp = await fetch(`/api/examples/task3/${demoCfg.prompt}`)
      const demoPrompt = promptResp.ok ? await promptResp.text() : ''

      // Set all inputs: image, Customize mode, 500px left+right, Middle position
      setImageFile(file)
      setImageUrl(url)
      setResultUrl(null)
      setInfo('')
      setTargetRes('Customize')
      setAlignment('Middle')
      setResizeOption('Full')
      setPadTop(0)
      setPadBottom(0)
      setPadLeft(500)
      setPadRight(500)
      setPrompt(demoPrompt)
      setInputTab(0)

      // Build form data manually for immediate inference
      setLoading(true)
      await fetch('/api/activate/task3', { method: 'POST' }).catch(() => { })

      const fd = new FormData()
      fd.append('image', file)
      fd.append('target_res', 'Customize')
      fd.append('custom_w', 1024)
      fd.append('custom_h', 1024)
      fd.append('alignment', 'Middle')
      fd.append('resize_option', 'Full')
      fd.append('custom_resize_pct', 100)
      fd.append('overlap_percentage', 10)
      fd.append('overlap_left', 10)
      fd.append('overlap_right', 10)
      fd.append('overlap_top', 10)
      fd.append('overlap_bottom', 10)
      fd.append('pad_left', 500)
      fd.append('pad_right', 500)
      fd.append('pad_top', 0)
      fd.append('pad_bottom', 0)
      fd.append('prompt', demoPrompt)
      fd.append('num_steps', numSteps)
      fd.append('sharpen', sharpen)
      fd.append('lora_scale', loraScale)

      const t0 = Date.now()
      const submitResp = await fetch('/api/task3/run', { method: 'POST', body: fd })
      if (!submitResp.ok) throw new Error(await submitResp.text())
      const { job_id } = await submitResp.json()

      while (true) {
        await new Promise(r => setTimeout(r, 10000))
        const statusResp = await fetch(`/api/job/${job_id}/status`)
        if (!statusResp.ok) throw new Error('Lost job status')
        const { status, error } = await statusResp.json()
        if (status === 'error') throw new Error(error || 'Inference failed')
        if (status === 'done') break
      }

      const resultResp = await fetch(`/api/job/${job_id}/result`)
      if (!resultResp.ok) throw new Error(await resultResp.text())
      const resultBlobUrl = URL.createObjectURL(await resultResp.blob())
      setResultUrl(resultBlobUrl)
      setInfo(`Hoàn thành trong ${((Date.now() - t0) / 1000).toFixed(1)}s · ${numSteps} steps`)
      setHistory((prev) => [{ url: resultBlobUrl }, ...prev.slice(0, 19)])
    } catch (err) {
      toast({ title: 'Demo Error', description: err.message, status: 'error', duration: 5000 })
    } finally {
      setLoading(false)
    }
  }

  const PillGroup = ({ options, value, onChange }) => (
    <HStack spacing={2} flexWrap="wrap">
      {options.map((opt) => (
        <Button
          key={opt} size="sm" fontSize="xs"
          variant={value === opt ? 'solid' : 'outline'}
          colorScheme={value === opt ? 'brand' : 'gray'}
          onClick={() => onChange(opt)}
        >
          {opt}
        </Button>
      ))}
    </HStack>
  )

  const locked = getLockedDirs(alignment)

  return (
    <Box>
      <Text color="gray.400" fontSize="sm" mb={5} lineHeight="tall">
        <Text as="span" fontWeight="700" color="gray.200">
          Mở rộng ảnh (Outpainting)
        </Text>
        {' — Ảnh gốc được đặt lên canvas SDXL, vùng màu đỏ trong preview là nơi model tự vẽ thêm.'}
      </Text>

      <DemoBox onDemo={handleDemo} loading={loading} />

      {/* ── Settings Panel ── */}
      <Box
        bg="whiteAlpha.50" borderRadius="lg" p={4} mb={5}
        border="1px solid" borderColor="whiteAlpha.100"
      >
        <Flex gap={6}>
          <Box flex={1}>
            <Text fontSize="xs" fontWeight="600" color="gray.400" mb={2}>
              📐 Kích thước đầu ra
            </Text>
            <PillGroup options={SDXL_BUCKETS} value={targetRes} onChange={setTargetRes} />
          </Box>
          <Box flex={1}>
            <Text fontSize="xs" fontWeight="600" color="gray.400" mb={2}>
              📍 Vị trí ảnh
            </Text>
            <Select
              size="sm" value={alignment}
              onChange={(e) => setAlignment(e.target.value)}
              bg="whiteAlpha.50" border="1px solid" borderColor="whiteAlpha.200"
            >
              {ALIGN_OPTS.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </Select>
          </Box>
          <Box flex={1}>
            <Text fontSize="xs" fontWeight="600" color="gray.400" mb={2}>
              🔲 Kích thước ảnh gốc
            </Text>
            <PillGroup options={RESIZE_OPTS} value={resizeOption} onChange={setResizeOption} />
          </Box>
        </Flex>
      </Box>

      {/* ── Advanced Settings ── */}
      <Accordion allowToggle mb={5}>
        <AccordionItem
          bg="whiteAlpha.50" borderRadius="lg"
          border="1px solid" borderColor="whiteAlpha.100"
        >
          <AccordionButton borderRadius="lg" _expanded={{ borderBottomRadius: 0 }}>
            <Text fontSize="sm" fontWeight="600" flex="1" textAlign="left">
              ⚙️ Cài đặt nâng cao
            </Text>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel>
            <SimpleGrid columns={2} spacing={6}>
              <Box>
                <Text fontSize="xs" fontWeight="600" color="gray.400" mb={3}>
                  Overlap theo hướng (%)
                </Text>
                <SimpleGrid columns={2} spacing={3}>
                  {[
                    { label: 'Trên ↑', val: overlapTop, set: setOverlapTop },
                    { label: 'Dưới ↓', val: overlapBottom, set: setOverlapBottom },
                    { label: 'Trái ←', val: overlapLeft, set: setOverlapLeft },
                    { label: 'Phải →', val: overlapRight, set: setOverlapRight },
                  ].map((s) => (
                    <Box key={s.label}>
                      <Flex justify="space-between" mb={1}>
                        <Text fontSize="xs" color="gray.400">{s.label}</Text>
                        <Text fontSize="xs" color="brand.300" fontWeight="600">{s.val}</Text>
                      </Flex>
                      <Slider
                        value={s.val} onChange={s.set}
                        min={0} max={50} size="sm"
                      >
                        <SliderTrack>
                          <SliderFilledTrack bg="brand.500" />
                        </SliderTrack>
                        <SliderThumb boxSize={3} />
                      </Slider>
                    </Box>
                  ))}
                </SimpleGrid>
              </Box>
              <Box>
                <Text fontSize="xs" fontWeight="600" color="gray.400" mb={3}>
                  Tham số inference
                </Text>
                {[
                  { label: 'Số bước', val: numSteps, set: setNumSteps, min: 10, max: 50, step: 5 },
                  { label: 'Sắc nét', val: sharpen, set: setSharpen, min: 0, max: 2, step: 0.1, fmt: (v) => v.toFixed(1) },
                  { label: 'LoRA Scale', val: loraScale, set: setLoraScale, min: 0, max: 1, step: 0.1, fmt: (v) => v.toFixed(1) },
                ].map((s) => (
                  <Box key={s.label} mb={3}>
                    <Flex justify="space-between" mb={1}>
                      <Text fontSize="xs" color="gray.400">{s.label}</Text>
                      <Text fontSize="xs" color="brand.300" fontWeight="600">
                        {s.fmt ? s.fmt(s.val) : s.val}
                      </Text>
                    </Flex>
                    <Slider
                      value={s.val} onChange={s.set}
                      min={s.min} max={s.max} step={s.step} size="sm"
                    >
                      <SliderTrack>
                        <SliderFilledTrack bg="brand.500" />
                      </SliderTrack>
                      <SliderThumb boxSize={3} />
                    </Slider>
                  </Box>
                ))}
              </Box>
            </SimpleGrid>
          </AccordionPanel>
        </AccordionItem>
      </Accordion>

      {/* ── Customize Padding ── */}
      {targetRes === 'Customize' && (
        <Box
          bg="whiteAlpha.50" borderRadius="lg" p={4} mb={5}
          border="1px solid" borderColor="whiteAlpha.100"
        >
          <Text fontSize="xs" fontWeight="600" color="gray.400" mb={3}>
            Mở rộng canvas tùy chỉnh (pixel)
          </Text>
          <SimpleGrid columns={4} spacing={3}>
            {[
              { label: 'Trên ↑', val: padTop, set: setPadTop, dir: 'top' },
              { label: 'Dưới ↓', val: padBottom, set: setPadBottom, dir: 'bottom' },
              { label: 'Trái ←', val: padLeft, set: setPadLeft, dir: 'left' },
              { label: 'Phải →', val: padRight, set: setPadRight, dir: 'right' },
            ].map((p) => {
              const isLocked = !!locked[p.dir]
              return (
                <Box key={p.label} opacity={isLocked ? 0.4 : 1}>
                  <Text fontSize="xs" color="gray.400" mb={1}>
                    {p.label} {isLocked && '🔒'}
                  </Text>
                  <NumberInput
                    size="sm" value={isLocked ? 0 : p.val}
                    onChange={(_, v) => p.set(v || 0)}
                    min={0} max={4096}
                    isDisabled={isLocked}
                  >
                    <NumberInputField
                      bg="whiteAlpha.50" border="1px solid"
                      borderColor="whiteAlpha.200"
                    />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </Box>
              )
            })}
          </SimpleGrid>
        </Box>
      )}

      {/* ── Images Row ── */}
      <SimpleGrid columns={2} spacing={6} mb={5}>
        {/* Left — Input (2 tabs: Ảnh đầu vào | Preview) */}
        <VStack align="stretch" spacing={0}>
          <Flex align="center" mb={2}>
            <Tabs
              index={inputTab} onChange={setInputTab}
              size="sm" variant="soft-rounded" colorScheme="brand"
            >
              <TabList>
                <Tab fontSize="xs">Ảnh đầu vào</Tab>
                <Tab fontSize="xs">Preview vùng mở rộng</Tab>
              </TabList>
            </Tabs>
          </Flex>
          <Box
            bg="whiteAlpha.50" borderRadius="lg" border="1px solid"
            borderColor="whiteAlpha.100" h={`${BOX_H}px`}
            display="flex" alignItems="center" justifyContent="center"
            overflow="hidden" position="relative"
            onDrop={handleDrop} onDragOver={handleDragOver}
            cursor={inputTab === 0 && !imageUrl ? 'pointer' : 'default'}
            onClick={() => inputTab === 0 && !imageUrl && fileRef.current?.click()}
            _hover={inputTab === 0 && !imageUrl ? { bg: 'whiteAlpha.100' } : {}}
            transition="all 0.2s"
          >
            {inputTab === 0 && (
              imageUrl ? (
                <ZoomableImage
                  src={imageUrl}
                  alt="Ảnh đầu vào"
                  caption="Ảnh đầu vào"
                  maxH={`${BOX_H}px`} objectFit="contain" w="100%"
                />
              ) : (
                <VStack spacing={2}>
                  <Text color="gray.500" fontSize="sm">📁 Click hoặc kéo thả ảnh vào đây</Text>
                  <Text color="gray.600" fontSize="xs">Hỗ trợ JPG, PNG, WebP</Text>
                </VStack>
              )
            )}
            {inputTab === 1 && (
              previewUrl ? (
                <ZoomableImage
                  src={previewUrl}
                  alt="Preview"
                  caption="Preview vùng mở rộng (vùng đỏ = nơi model tự vẽ thêm)"
                  maxH={`${BOX_H}px`} objectFit="contain" w="100%"
                />
              ) : (
                <Text color="gray.500" fontSize="sm">Upload ảnh để xem preview</Text>
              )
            )}
          </Box>
          {imageUrl && (
            <HStack mt={2}>
              <Button
                size="xs" variant="outline" alignSelf="flex-start"
                onClick={() => fileRef.current?.click()}
              >
                Đổi ảnh
              </Button>
              <Button
                size="xs" variant="outline" colorScheme="red" alignSelf="flex-start"
                onClick={handleClearImage}
              >
                Xóa ảnh
              </Button>
            </HStack>
          )}
          <input
            ref={fileRef} type="file" accept="image/*"
            style={{ display: 'none' }} onChange={handleFileChange}
          />
        </VStack>

        {/* Right — Result */}
        <VStack align="stretch" spacing={0}>
          <Text fontWeight="600" fontSize="sm" color="gray.300" mb={2} h="32px" lineHeight="32px">
            Kết quả
          </Text>
          <Box
            bg="whiteAlpha.50" borderRadius="lg" border="1px solid"
            borderColor="whiteAlpha.100" h={`${BOX_H}px`} position="relative"
            display="flex" alignItems="center" justifyContent="center"
            overflow="hidden" role="group"
          >
            {loading && (
              <Center position="absolute" inset={0} bg="blackAlpha.700" zIndex={2}>
                <VStack>
                  <Spinner color="brand.400" size="lg" thickness="3px" />
                  <Text fontSize="sm" mt={2}>Đang xử lý...</Text>
                </VStack>
              </Center>
            )}
            {resultUrl ? (
              <>
                <ZoomableImage
                  src={resultUrl}
                  alt="Kết quả"
                  caption="Kết quả Outpainting"
                  maxH={`${BOX_H}px`} objectFit="contain" w="100%"
                />
                {/* Download button — top-left, visible on hover */}
                <Box
                  as="a" href={resultUrl} download="outpainting_result.png"
                  position="absolute" top="8px" left="8px" zIndex={4}
                  opacity={0} transition="opacity 0.15s"
                  _groupHover={{ opacity: 1 }}
                  w="30px" h="30px"
                  borderRadius="md"
                  bg="blackAlpha.700"
                  display="flex" alignItems="center" justifyContent="center"
                  color="white"
                  _hover={{ bg: 'blackAlpha.900', opacity: 1 }}
                  title="Tải xuống"
                  onClick={(e) => e.stopPropagation()}
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </Box>
              </>
            ) : (
              !loading && <Text color="gray.500" fontSize="sm">Kết quả sẽ hiện ở đây</Text>
            )}
            {info && (
              <Box
                position="absolute" bottom={3} right={3}
                bg="blackAlpha.600" borderRadius="md"
                px={2} py={1}
              >
                <Text color="green.300" fontSize="xs">
                  {info}
                </Text>
              </Box>
            )}
          </Box>
        </VStack>
      </SimpleGrid>

      {/* ── Prompt + Generate ── */}
      <Flex gap={4} align="end" mb={5}>
        <Box flex={1}>
          <Text fontSize="xs" fontWeight="600" color="gray.400" mb={1}>
            Prompt mô tả nội dung mở rộng (tuỳ chọn)
          </Text>
          <Textarea
            size="sm" rows={2} bg="whiteAlpha.50"
            border="1px solid" borderColor="whiteAlpha.200"
            placeholder="Để trống để dùng prompt mặc định..."
            value={prompt} onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.stopPropagation()}
            _focus={{ borderColor: 'brand.400' }}
          />
        </Box>
        <Button
          colorScheme="brand" size="lg" px={8}
          onClick={handleRun}
          isDisabled={loading || !imageFile}
          isLoading={loading} loadingText="Đang xử lý..."
          flexShrink={0}
        >
          🚀 Generate
        </Button>
      </Flex>

      {/* ── History ── */}
      {history.length > 0 && (
        <Accordion defaultIndex={[0]} allowToggle>
          <AccordionItem
            bg="whiteAlpha.50" borderRadius="lg"
            border="1px solid" borderColor="whiteAlpha.100"
          >
            <AccordionButton borderRadius="lg">
              <Text fontSize="sm" fontWeight="600" flex="1" textAlign="left">
                📋 Lịch sử xử lý ({history.length})
              </Text>
              <Button
                size="xs" variant="ghost" colorScheme="red" mr={2}
                onClick={(e) => { e.stopPropagation(); setHistory([]) }}
              >
                Xóa
              </Button>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel>
              <SimpleGrid columns={4} spacing={2}>
                {history.map((h, i) => (
                  <ZoomableImage
                    key={i} src={h.url}
                    alt={`Lịch sử ${i + 1}`}
                    caption={`Kết quả #${history.length - i}`}
                    borderRadius="md"
                    objectFit="cover" aspectRatio={1}
                    cursor="zoom-in" border="1px solid" borderColor="whiteAlpha.100"
                    _hover={{ borderColor: 'brand.400' }}
                    transition="all 0.2s"
                  />
                ))}
              </SimpleGrid>
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
      )}
    </Box>
  )
})

export default Task3Tab
