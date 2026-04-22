import React, { useState, useRef, useCallback, useEffect, forwardRef, useImperativeHandle } from 'react'
import {
  Box, Button, SimpleGrid, Text, VStack, HStack, Center, Flex,
  Spinner, Textarea, useToast,
  Slider, SliderTrack, SliderFilledTrack, SliderThumb,
  Accordion, AccordionItem, AccordionButton, AccordionPanel, AccordionIcon,
} from '@chakra-ui/react'
import { ZoomableImage, LightboxModal } from './ImageLightbox.jsx'
import MaskEditor from './MaskEditor.jsx'
import DemoBox from './DemoBox.jsx'

const MODES = ['Add', 'Delete', 'Replace']

const DEFAULT_NEG_ADD =
  'additional objects, extra items, unwanted objects, multiple objects, cluttered, busy background, unrelated decoration, extra furniture, low quality, blurry, distorted, deformed, artifacts, 3d render, CGI, plastic texture, flat shading, floating, wrong perspective, impossible geometry, cartoon, anime, painting, sketch, text, watermark, logo'
const DEFAULT_NEG_DELETE =
  'object, furniture, item, person, blurry, smudge, artifacts, distorted texture, mismatched pattern, text, watermark, low quality, messy, floating debris, ghosting'

const DEFAULTS = {
  Add: { strength: 1.0, cfg: 12.0, neg: DEFAULT_NEG_ADD, steps: 20 },
  Replace: { strength: 1.0, cfg: 12.0, neg: DEFAULT_NEG_ADD, steps: 20 },
  Delete: { strength: 0.99, cfg: 20.0, neg: DEFAULT_NEG_DELETE, steps: 20 },
}

const BOX_H = 500  // matching height for input & output boxes

const Task2Tab = forwardRef(function Task2Tab({ onSendToTask3 }, ref) {
  const [imageFile, setImageFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [info, setInfo] = useState('')
  const [loading, setLoading] = useState(false)
  const [enhancing, setEnhancing] = useState(false)
  const [lightboxOpen, setLightboxOpen] = useState(false)

  const [mode, setMode] = useState('Add')
  const [steps, setSteps] = useState(50)
  const [strength, setStrength] = useState(1.0)
  const [cfg, setCfg] = useState(12.0)
  const [cnScale, setCnScale] = useState(0.3)
  const [prompt, setPrompt] = useState('')
  const [negPrompt, setNegPrompt] = useState(DEFAULT_NEG_ADD)

  const editorRef = useRef(null)
  const fileRef = useRef(null)
  const toast = useToast()

  useEffect(() => {
    const d = DEFAULTS[mode]
    setStrength(d.strength)
    setCfg(d.cfg)
    setNegPrompt(d.neg)
    setSteps(d.steps)
  }, [mode])

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImageFile(file)
    setImageUrl(URL.createObjectURL(file))
    setResult(null)
    setInfo('')
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const file = Array.from(e.dataTransfer.files).find((f) => f.type.startsWith('image/'))
    if (file) {
      setImageFile(file)
      setImageUrl(URL.createObjectURL(file))
      setResult(null)
      setInfo('')
    }
  }, [])

  const handleClearImage = useCallback(() => {
    setImageFile(null)
    setImageUrl(null)
    setResult(null)
    setInfo('')
  }, [])

  const handleEnhance = async () => {
    if (!prompt.trim()) return
    setEnhancing(true)
    try {
      const fd = new FormData()
      fd.append('prompt', prompt)
      const resp = await fetch('/api/task2/enhance-prompt', { method: 'POST', body: fd })
      if (!resp.ok) throw new Error(await resp.text())
      const { job_id } = await resp.json()

      // Poll for completion
      while (true) {
        const statusResp = await fetch(`/api/job/${job_id}/status`)
        if (!statusResp.ok) throw new Error('Lost enhance job')
        const data = await statusResp.json()
        if (data.status === 'done') {
          const parsed = JSON.parse(data.text || '{}')
          if (parsed.positive) setPrompt(parsed.positive)
          if (parsed.negative) setNegPrompt(parsed.negative)
          toast({ title: 'Prompt enhanced', status: 'success', duration: 2000 })
          break
        }
        if (data.status === 'error') throw new Error(data.error || 'Enhance failed')
        await new Promise((r) => setTimeout(r, 1500))
      }
    } catch (err) {
      toast({ title: 'Enhance failed', description: err.message, status: 'error', duration: 3000 })
    } finally {
      setEnhancing(false)
    }
  }

  const loadExternalImage = useCallback(async (blobUrl) => {
    try {
      const resp = await fetch(blobUrl)
      const blob = await resp.blob()
      const file = new File([blob], 'image.png', { type: blob.type })
      setImageFile(file)
      setImageUrl(URL.createObjectURL(blob))
      setResult(null)
      setInfo('')
    } catch { /* ignore */ }
  }, [])

  useImperativeHandle(ref, () => ({ loadExternalImage }), [loadExternalImage])

  const handleRun = async () => {
    if (!imageFile) return
    const maskBlob = await editorRef.current?.getMaskBlob()
    if (!maskBlob) return
    setLoading(true)
    setResult(null)
    setInfo('')
    try {
      await fetch('/api/activate/task2', { method: 'POST' }).catch(() => { })

      const fd = new FormData()
      fd.append('image', imageFile)
      fd.append('mask', maskBlob, 'mask.png')
      fd.append('prompt', prompt)
      fd.append('negative_prompt', negPrompt)
      fd.append('task_type', mode)
      fd.append('steps', steps)
      fd.append('strength', strength)
      fd.append('guidance_scale', cfg)
      fd.append('cn_scale', cnScale)

      const submitResp = await fetch('/api/task2/run', { method: 'POST', body: fd })
      if (!submitResp.ok) throw new Error(await submitResp.text())
      const { job_id } = await submitResp.json()

      while (true) {
        await new Promise((r) => setTimeout(r, 10000))
        const statusResp = await fetch(`/api/job/${job_id}/status`)
        if (!statusResp.ok) throw new Error('Lost job status')
        const { status, error } = await statusResp.json()
        if (status === 'error') throw new Error(error || 'Inference failed')
        if (status === 'done') break
      }

      const resultResp = await fetch(`/api/job/${job_id}/result`)
      if (!resultResp.ok) throw new Error(await resultResp.text())
      const blob = await resultResp.blob()
      try {
        setInfo(decodeURIComponent(resultResp.headers.get('X-Info') || ''))
      } catch (_) {
        setInfo('')
      }
      setResult(URL.createObjectURL(blob))
    } catch (err) {
      toast({ title: 'Error', description: err.message, status: 'error', duration: 5000 })
    } finally {
      setLoading(false)
    }
  }

  const DEMO_CONFIGS = {
    1: { image: '7.jpg', mask: '7.png', prompt: '7.txt' },
    2: { image: '10.jpg', mask: '10.png', prompt: '10.txt' },
  }

  const handleDemo = async (n) => {
    const cfg = DEMO_CONFIGS[n]
    if (!cfg) return
    try {
      // Fetch image
      const imgResp = await fetch(`/api/examples/task2/${cfg.image}`)
      if (!imgResp.ok) throw new Error('Failed to load demo image')
      const imgBlob = await imgResp.blob()
      const file = new File([imgBlob], cfg.image, { type: imgBlob.type })
      const url = URL.createObjectURL(imgBlob)
      setImageFile(file)
      setImageUrl(url)
      setResult(null)
      setInfo('')

      // Fetch prompt
      let demoPrompt = ''
      const promptResp = await fetch(`/api/examples/task2/${cfg.prompt}`)
      if (promptResp.ok) {
        demoPrompt = await promptResp.text()
        setPrompt(demoPrompt)
      }

      // Set mode to Add for demo
      setMode('Add')

      // Wait for MaskEditor to load the image, then apply the mask
      const maskUrl = `/api/examples/task2/${cfg.mask}`
      const waitForEditor = () => new Promise((resolve) => {
        const check = () => {
          if (editorRef.current?.loadMask) {
            resolve()
          } else {
            setTimeout(check, 100)
          }
        }
        // Give time for image to load in MaskEditor
        setTimeout(check, 500)
      })
      await waitForEditor()
      // Allow image to fully render in MaskEditor canvas
      await new Promise(r => setTimeout(r, 800))
      await editorRef.current.loadMask(maskUrl)

      // Now run inference
      const maskBlob = await editorRef.current.getMaskBlob()
      if (!maskBlob) throw new Error('Failed to get mask')
      setLoading(true)
      await fetch('/api/activate/task2', { method: 'POST' }).catch(() => { })

      const fd = new FormData()
      fd.append('image', file)
      fd.append('mask', maskBlob, 'mask.png')
      fd.append('prompt', demoPrompt)
      fd.append('negative_prompt', DEFAULT_NEG_ADD)
      fd.append('task_type', 'Add')
      fd.append('steps', 50)
      fd.append('strength', 1.0)
      fd.append('guidance_scale', 12.0)
      fd.append('cn_scale', 0.3)

      const submitResp = await fetch('/api/task2/run', { method: 'POST', body: fd })
      if (!submitResp.ok) throw new Error(await submitResp.text())
      const { job_id } = await submitResp.json()

      while (true) {
        await new Promise((r) => setTimeout(r, 10000))
        const statusResp = await fetch(`/api/job/${job_id}/status`)
        if (!statusResp.ok) throw new Error('Lost job status')
        const { status, error } = await statusResp.json()
        if (status === 'error') throw new Error(error || 'Inference failed')
        if (status === 'done') break
      }

      const resultResp = await fetch(`/api/job/${job_id}/result`)
      if (!resultResp.ok) throw new Error(await resultResp.text())
      setInfo(decodeURIComponent(resultResp.headers.get('X-Info') || ''))
      setResult(URL.createObjectURL(await resultResp.blob()))
    } catch (err) {
      toast({ title: 'Demo Error', description: err.message, status: 'error', duration: 5000 })
    } finally {
      setLoading(false)
    }
  }

  const SLIDERS = [
    { label: 'Steps', val: steps, set: setSteps, min: 10, max: 50, step: 1 },
    { label: 'Strength', val: strength, set: setStrength, min: 0.1, max: 1.0, step: 0.05, fmt: (v) => v.toFixed(2) },
    { label: 'CFG', val: cfg, set: setCfg, min: 1, max: 20, step: 0.5, fmt: (v) => v.toFixed(1) },
    { label: 'CN Scale', val: cnScale, set: setCnScale, min: 0, max: 1, step: 0.1, fmt: (v) => v.toFixed(1) },
  ]

  return (
    <Box>
      <Text color="gray.400" fontSize="sm" mb={5} lineHeight="tall">
        <Text as="span" fontWeight="700" color="gray.200">
          Chỉnh sửa ảnh (Inpainting)
        </Text>
        {' — Vẽ mask trên vùng muốn sửa/điền, sau đó chọn chế độ Add, Delete hoặc Replace.'}
      </Text>

      <DemoBox onDemo={handleDemo} loading={loading} />

      {/* Settings Panel */}
      <Box
        bg="whiteAlpha.50" borderRadius="lg" p={4} mb={5}
        border="1px solid" borderColor="whiteAlpha.100"
      >
        <SimpleGrid columns={5} spacing={4} alignItems="end">
          <Box>
            <Text fontSize="xs" fontWeight="600" color="gray.400" mb={2}>Chế độ</Text>
            <HStack spacing={2}>
              {MODES.map((m) => (
                <Button
                  key={m} size="sm" fontSize="xs"
                  variant={mode === m ? 'solid' : 'outline'}
                  colorScheme={mode === m ? 'brand' : 'gray'}
                  onClick={() => setMode(m)}
                >
                  {m}
                </Button>
              ))}
            </HStack>
          </Box>
          {SLIDERS.map((s) => (
            <Box key={s.label}>
              <Flex justify="space-between" mb={1}>
                <Text fontSize="xs" fontWeight="600" color="gray.400">{s.label}</Text>
                <Text fontSize="xs" color="brand.300" fontWeight="600">{s.fmt ? s.fmt(s.val) : s.val}</Text>
              </Flex>
              <Slider value={s.val} onChange={s.set} min={s.min} max={s.max} step={s.step} size="sm">
                <SliderTrack><SliderFilledTrack bg="brand.500" /></SliderTrack>
                <SliderThumb boxSize={3} />
              </Slider>
            </Box>
          ))}
        </SimpleGrid>
      </Box>

      {/* Main Area — two equal-height columns */}
      <SimpleGrid columns={2} spacing={6}>
        {/* Left — Mask Editor */}
        <VStack align="stretch" spacing={3}>
          <Text fontWeight="600" fontSize="sm" color="gray.300">Vẽ mask</Text>
          <Box
            bg="whiteAlpha.50" borderRadius="lg" border="1px solid"
            borderColor="whiteAlpha.100" p={3} h={`${BOX_H}px`}
            position="relative"
          >
            {imageUrl && (
              <Box
                as="button"
                position="absolute" top="8px" right="8px" zIndex={5}
                w="26px" h="26px" borderRadius="full"
                bg="blackAlpha.600" border="none" outline="none"
                display="flex" alignItems="center" justifyContent="center"
                cursor="pointer" color="white"
                _hover={{ bg: 'red.500' }}
                onClick={handleClearImage}
                title="Xóa ảnh"
              >
                <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="1" y1="1" x2="9" y2="9" />
                  <line x1="9" y1="1" x2="1" y2="9" />
                </svg>
              </Box>
            )}
            {imageUrl ? (
              <MaskEditor
                ref={editorRef}
                imageUrl={imageUrl}
                height={BOX_H - 24}
                onOpenLightbox={() => setLightboxOpen(true)}
              />
            ) : (
              <Center
                h="100%" cursor="pointer"
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                _hover={{ bg: 'whiteAlpha.100' }}
                borderRadius="md" transition="all 0.2s"
              >
                <VStack spacing={1}>
                  <Text color="gray.500" fontSize="sm">📁 Click hoặc kéo thả để upload ảnh</Text>
                  <Text color="gray.600" fontSize="xs">Hỗ trợ JPG, PNG, WebP</Text>
                </VStack>
              </Center>
            )}
            <input
              ref={fileRef} type="file" accept="image/*"
              style={{ display: 'none' }} onChange={handleFileChange}
            />
          </Box>

          {/* Prompt */}
          <Box>
            <Text fontSize="xs" fontWeight="600" color="gray.400" mb={1}>Prompt yêu cầu</Text>
            <Textarea
              size="sm" rows={3} bg="whiteAlpha.50"
              border="1px solid" borderColor="whiteAlpha.200"
              placeholder="Ví dụ: a modern lamp, warm lighting, ..."
              value={prompt} onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.stopPropagation()}
              _focus={{ borderColor: 'brand.400' }}
            />
          </Box>
          <Button
            size="sm" variant="outline" colorScheme="brand"
            onClick={handleEnhance}
            isDisabled={enhancing || !prompt.trim()}
            isLoading={enhancing} loadingText="Đang enhance..."
            alignSelf="flex-start"
          >
            ✨ Enhance Prompt
          </Button>

          {/* Negative Prompt Accordion */}
          <Accordion allowToggle>
            <AccordionItem border="none">
              <AccordionButton px={0} _hover={{ bg: 'transparent' }}>
                <Text fontSize="xs" fontWeight="600" color="gray.400" flex="1" textAlign="left">
                  ⚙️ Negative Prompt
                </Text>
                <AccordionIcon />
              </AccordionButton>
              <AccordionPanel px={0} pb={0}>
                <Textarea
                  size="sm" rows={4} bg="whiteAlpha.50"
                  border="1px solid" borderColor="whiteAlpha.200"
                  value={negPrompt} onChange={(e) => setNegPrompt(e.target.value)}
                  onKeyDown={(e) => e.stopPropagation()}
                  _focus={{ borderColor: 'brand.400' }}
                />
              </AccordionPanel>
            </AccordionItem>
          </Accordion>
        </VStack>

        {/* Right — Result */}
        <VStack align="stretch" spacing={4}>
          <Text fontWeight="600" fontSize="sm" color="gray.300">Kết quả</Text>
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
            {result ? (
              <>
                <ZoomableImage
                  src={result}
                  alt="Kết quả inpainting"
                  caption="Kết quả Inpainting & Editing"
                  downloadSrc={result}
                  downloadName="result.png"
                  maxH={`${BOX_H}px`} objectFit="contain" w="100%"
                />
                {/* Download button — top-left of result frame, visible on hover */}
                <Box
                  as="a" href={result} download="result.png"
                  position="absolute" top="8px" left="8px" zIndex={4}
                  opacity={0} transition="opacity 0.15s"
                  _groupHover={{ opacity: 1 }}
                  w="30px" h="30px" borderRadius="md" bg="blackAlpha.700"
                  display="flex" alignItems="center" justifyContent="center"
                  color="white" _hover={{ bg: 'blackAlpha.900', opacity: 1 }}
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
              <Text color="gray.500" fontSize="sm">Kết quả sẽ hiện ở đây</Text>
            )}
            {info && (
              <Box
                position="absolute" bottom={3} right={3}
                bg="blackAlpha.600" borderRadius="md"
                px={2} py={1}
              >
                <Text color="green.300" fontSize="xs">{info}</Text>
              </Box>
            )}
          </Box>

          {result && onSendToTask3 && (
            <HStack spacing={2} pt={1}>
              <Button size="sm" variant="outline" colorScheme="purple"
                onClick={() => onSendToTask3(result)}
              >
                → Dùng cho Outpainting
              </Button>
            </HStack>
          )}

          <Button
            colorScheme="brand" size="lg" w="full"
            onClick={handleRun}
            isDisabled={loading || !imageFile}
            isLoading={loading} loadingText="Đang xử lý..."
          >
            🚀 Xử lý
          </Button>
        </VStack>
      </SimpleGrid>

      {/* Lightbox for full-size preview (when no mask tool selected and user clicks canvas) */}
      {imageUrl && (
        <LightboxModal
          isOpen={lightboxOpen}
          onClose={() => setLightboxOpen(false)}
          images={[{ src: imageUrl, caption: 'Ảnh gốc' }]}
          initialIndex={0}
        />
      )}
    </Box>
  )
})

export default Task2Tab
