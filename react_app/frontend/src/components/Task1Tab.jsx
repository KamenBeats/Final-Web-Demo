import React, { useState, useRef, useCallback } from 'react'
import {
  Box, Button, SimpleGrid, Text, VStack, HStack, Center, Flex,
  Spinner, Badge, Switch, FormControl, FormLabel, useToast,
  Slider, SliderTrack, SliderFilledTrack, SliderThumb,
} from '@chakra-ui/react'
import { ZoomableImage } from './ImageLightbox.jsx'
import DemoBox from './DemoBox.jsx'

const DEMO_CONFIGS = {
  1: { images: ['1/1.jpg', '1/2.jpg', '1/3.jpg', '1/4.jpg', '1/5.jpg'] },
  2: { images: ['2/1.jpg', '2/2.jpg', '2/3.jpg'] },
}

export default function Task1Tab({ onSendToTask2, onSendToTask3 }) {
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [exposureLabels, setExposureLabels] = useState([])  // ['EV -2','EV -1','EV +1'] or []
  const [result, setResult] = useState(null)
  const [info, setInfo] = useState('')
  const [loading, setLoading] = useState(false)
  const [brightness, setBrightness] = useState(100)  // 100 = chuẩn model, map 1-100 → 0.01-1.0
  const [generatingExposures, setGeneratingExposures] = useState(false)
  const fileRef = useRef(null)
  const exposureFileRef = useRef(null)
  const toast = useToast()

  // Append new files instead of replacing
  const handleFiles = useCallback((e) => {
    const newFiles = Array.from(e.target.files)
    if (!newFiles.length) return
    const newPreviews = newFiles.map((f) => URL.createObjectURL(f))
    setFiles((prev) => [...prev, ...newFiles])
    setPreviews((prev) => [...prev, ...newPreviews])
    setExposureLabels([])
    // Reset so same file can be re-added
    e.target.value = ''
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const dropped = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith('image/'))
    if (dropped.length) {
      const newPreviews = dropped.map((f) => URL.createObjectURL(f))
      setFiles((prev) => [...prev, ...dropped])
      setPreviews((prev) => [...prev, ...newPreviews])
      setExposureLabels([])
    }
  }, [])

  const handleDeleteImage = useCallback((deleteIdx) => {
    setPreviews((prev) => {
      URL.revokeObjectURL(prev[deleteIdx])
      return prev.filter((_, i) => i !== deleteIdx)
    })
    setFiles((prev) => prev.filter((_, i) => i !== deleteIdx))
    setExposureLabels([])
  }, [])

  // Generate 3 exposure-bracketed images from a single source image (no inference)
  const handleGenerateExposures = useCallback(async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    setGeneratingExposures(true)
    try {
      const img = new window.Image()
      const url = URL.createObjectURL(file)
      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = reject
        img.src = url
      })
      URL.revokeObjectURL(url)

      // EV stops: -2, -1, +1 (3 ảnh, không gồm EV 0 = bản gốc)
      const evStops = [-2, -1, 1]

      const results = await Promise.all(evStops.map((ev) => new Promise((resolve) => {
        const factor = Math.pow(2, ev)
        const canvas = document.createElement('canvas')
        canvas.width = img.width
        canvas.height = img.height
        const ctx = canvas.getContext('2d')
        ctx.drawImage(img, 0, 0)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const d = imageData.data
        for (let i = 0; i < d.length; i += 4) {
          // Gamma-correct exposure: decode sRGB → linear → scale → re-encode
          for (let c = 0; c < 3; c++) {
            const norm = d[i + c] / 255
            const linear = Math.pow(norm, 2.2) * factor
            d[i + c] = Math.round(Math.min(1, Math.pow(linear, 1 / 2.2)) * 255)
          }
        }
        ctx.putImageData(imageData, 0, 0)
        const previewUrl = canvas.toDataURL('image/png')
        canvas.toBlob((blob) => {
          resolve({
            file: new File([blob], `ev${ev > 0 ? '+' : ''}${ev}.png`, { type: 'image/png' }),
            preview: previewUrl,
          })
        }, 'image/png')
      })))

      // Replace current inputs with the 3 generated images
      previews.forEach((p) => URL.revokeObjectURL(p))
      setFiles(results.map((r) => r.file))
      setPreviews(results.map((r) => r.preview))
      setExposureLabels(evStops.map((ev) => `EV ${ev > 0 ? '+' : ''}${ev}`))
      setResult(null)
      setInfo('')
      toast({
        title: 'Đã tạo 3 ảnh đa phơi sáng',
        description: 'EV −2, EV −1, EV +1 — bấm Ghép & Xử lý để fusion.',
        status: 'success',
        duration: 4000,
      })
    } catch (err) {
      toast({ title: 'Lỗi tạo ảnh', description: err.message, status: 'error', duration: 5000 })
    } finally {
      setGeneratingExposures(false)
    }
  }, [previews, toast])

  const handleRun = useCallback(async (demoFiles = null) => {
    const filesToUse = demoFiles || files
    if (!filesToUse.length) return
    setLoading(true)
    setResult(null)
    setInfo('')
    try {
      await fetch('/api/activate/task1', { method: 'POST' }).catch(() => { })

      const fd = new FormData()
      filesToUse.forEach((f) => fd.append('files', f))
      // Phase 2 (ToneNet) luôn bật
      fd.append('apply_phase2', true)
      fd.append('align', true)
      // Map 1–100% → 0.01–1.0 (100% = chuẩn model)
      fd.append('brightness', (brightness / 100).toFixed(2))
      const submitResp = await fetch('/api/task1/run', { method: 'POST', body: fd })
      if (!submitResp.ok) throw new Error(await submitResp.text())
      const { job_id } = await submitResp.json()

      // Poll for completion (same pattern as Task2/Task3)
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const statusResp = await fetch(`/api/job/${job_id}/status`)
        if (!statusResp.ok) throw new Error('Lost job status')
        const { status, error } = await statusResp.json()
        if (status === 'done') break
        if (status === 'error') throw new Error(error)
        await new Promise((r) => setTimeout(r, 1500))
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
  }, [files, brightness, toast])

  const handleDemo = useCallback(async (n) => {
    const cfg = DEMO_CONFIGS[n]
    if (!cfg) return
    setLoading(true)
    setResult(null)
    setInfo('')
    try {
      // Set brightness based on demo: demo 1 = 100%, demo 2 = 80%
      if (n === 1) setBrightness(100)
      else if (n === 2) setBrightness(80)

      // Fetch all demo images as File objects
      const fetched = await Promise.all(
        cfg.images.map(async (path) => {
          const resp = await fetch(`/api/examples/task1/${path}`)
          if (!resp.ok) throw new Error(`Failed to load ${path}`)
          const blob = await resp.blob()
          return new File([blob], path.split('/').pop(), { type: blob.type })
        })
      )
      // Update previews
      const newPreviews = fetched.map((f) => URL.createObjectURL(f))
      setFiles(fetched)
      setPreviews(newPreviews)
      setExposureLabels([])
      // Run inference with the fetched files directly
      await handleRun(fetched)
    } catch (err) {
      toast({ title: 'Demo Error', description: err.message, status: 'error', duration: 5000 })
      setLoading(false)
    }
  }, [handleRun, toast])

  // Gallery array for lightbox navigation
  const gallery = previews.map((src, i) => ({
    src,
    caption: `Ảnh đầu vào ${i + 1} / ${previews.length}`,
  }))

  return (
    <Box>
      <Text color="gray.400" fontSize="sm" mb={5} lineHeight="tall">
        <Text as="span" fontWeight="700" color="gray.200">
          Multi-Exposure Fusion (UFRetinex-MEF-ToneNet)
        </Text>
        {' — Tải 2+ ảnh với các mức độ sáng khác nhau để ghép thành 1 ảnh cân bằng ánh sáng. Hoặc 1 ảnh để tăng cường ánh sáng.'}
      </Text>

      <DemoBox onDemo={handleDemo} loading={loading} />

      <SimpleGrid columns={2} spacing={6} alignItems="start">
        {/* Left — Upload & Preview */}
        <VStack align="stretch" spacing={4}>
          <Text fontWeight="600" fontSize="sm" color="gray.300">
            Ảnh đầu vào {previews.length > 0 && <Text as="span" color="gray.500">({previews.length} ảnh)</Text>}
          </Text>

          {/* Preview grid — same fixed height as result box */}
          <Box
            bg="whiteAlpha.50" borderRadius="lg" border="1px solid"
            borderColor="whiteAlpha.100" p={3}
            h="350px" overflowY="auto"
          >
            {previews.length > 0 ? (
              <SimpleGrid columns={3} spacing={2}>
                {previews.map((src, i) => (
                  <Box
                    key={i} position="relative" role="group"
                    borderRadius="md" overflow="hidden"
                  >
                    {/* Dark overlay on image hover */}
                    <Box
                      position="absolute" inset={0} zIndex={1}
                      bg="blackAlpha.500"
                      opacity={0} transition="opacity 0.15s"
                      pointerEvents="none"
                      _groupHover={{ opacity: 1 }}
                    />
                    <ZoomableImage
                      src={src}
                      alt={`Ảnh đầu vào ${i + 1}`}
                      gallery={gallery}
                      galleryIndex={i}
                      borderRadius="md"
                      objectFit="cover"
                      w="100%"
                      aspectRatio={1}
                      display="block"
                    />
                    {/* EV label badge */}
                    {exposureLabels[i] && (
                      <Box
                        position="absolute" bottom="4px" left="50%" zIndex={2}
                        transform="translateX(-50%)"
                        bg="blackAlpha.700" borderRadius="sm" px={1}
                        pointerEvents="none"
                      >
                        <Text fontSize="9px" fontWeight="700" color={
                          exposureLabels[i].includes('+') ? 'orange.300' : 'blue.300'
                        }>
                          {exposureLabels[i]}
                        </Text>
                      </Box>
                    )}
                    {/* Delete button */}
                    <Box
                      as="button"
                      position="absolute" top="5px" right="5px" zIndex={3}
                      opacity={0} transition="opacity 0.15s"
                      _groupHover={{ opacity: 1 }}
                      w="22px" h="22px"
                      borderRadius="full"
                      bg="blackAlpha.600"
                      display="flex" alignItems="center" justifyContent="center"
                      cursor="pointer" color="white"
                      border="none" outline="none"
                      _hover={{ bg: 'blackAlpha.900', opacity: 1 }}
                      onClick={(e) => { e.stopPropagation(); handleDeleteImage(i) }}
                    >
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                        <line x1="1" y1="1" x2="9" y2="9" />
                        <line x1="9" y1="1" x2="1" y2="9" />
                      </svg>
                    </Box>
                  </Box>
                ))}
              </SimpleGrid>
            ) : (
              <Center h="100%" color="gray.500" fontSize="sm">
                Chưa có ảnh
              </Center>
            )}
          </Box>

          {/* Drop zone */}
          <Box
            p={4} border="2px dashed" borderColor="whiteAlpha.200" borderRadius="lg"
            bg="whiteAlpha.50" cursor="pointer" textAlign="center"
            _hover={{ borderColor: 'brand.400', bg: 'whiteAlpha.100' }}
            transition="all 0.2s"
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <input
              ref={fileRef} type="file" accept="image/*" multiple
              onChange={handleFiles} style={{ display: 'none' }}
            />
            <Text color="gray.400" fontSize="sm">
              📁 Kéo thả hoặc click để thêm ảnh
            </Text>
            <Text color="gray.500" fontSize="xs" mt={1}>
              2+ ảnh cùng kích thước được khuyến nghị
            </Text>
          </Box>

          {/* Tạo 3 ảnh đa phơi sáng từ 1 ảnh */}
          <Box>
            <input
              ref={exposureFileRef} type="file" accept="image/*"
              onChange={handleGenerateExposures} style={{ display: 'none' }}
            />
            <Button
              w="full" size="sm" variant="outline" colorScheme="orange"
              isLoading={generatingExposures}
              loadingText="Đang tạo ảnh..."
              onClick={() => exposureFileRef.current?.click()}
              leftIcon={
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" />
                  <line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" />
                  <line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
              }
            >
              Tạo 3 ảnh đa phơi sáng từ 1 ảnh
            </Button>
            <Text color="gray.600" fontSize="xs" mt={1} textAlign="center">
              Tự động tạo EV −2, −1, +1 (không inference ngay)
            </Text>
          </Box>
        </VStack>

        {/* Right — Result */}
        <VStack align="stretch" spacing={4}>
          <Text fontWeight="600" fontSize="sm" color="gray.300">
            Kết quả
          </Text>
          <Box
            bg="whiteAlpha.50" borderRadius="lg" border="1px solid"
            borderColor="whiteAlpha.100" h="350px" position="relative"
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
                  alt="Kết quả"
                  caption="Kết quả Multi-Exposure Fusion"
                  downloadSrc={result}
                  downloadName="result.png"
                  maxH="350px" objectFit="contain" w="100%"
                />
                {/* Download button — top-left of result frame, visible on hover */}
                <Box
                  as="a" href={result} download="result.png"
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
              <Text color="gray.500" fontSize="sm">Kết quả sẽ hiện ở đây</Text>
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

          {result && (
            <HStack spacing={2} pt={1} flexWrap="wrap">
              {onSendToTask2 && (
                <Button size="sm" variant="outline" colorScheme="teal"
                  onClick={() => onSendToTask2(result)}
                >
                  → Dùng cho Inpainting
                </Button>
              )}
              {onSendToTask3 && (
                <Button size="sm" variant="outline" colorScheme="purple"
                  onClick={() => onSendToTask3(result)}
                >
                  → Dùng cho Outpainting
                </Button>
              )}
            </HStack>
          )}

          {/* Tăng cường ánh sáng & màu sắc */}
          <Box px={1} pt={2}>
            <Flex justify="space-between" mb={1}>
              <Text fontSize="xs" color="gray.400">☀️ Độ tăng cường ánh sáng & màu sắc</Text>
              <Text fontSize="xs" color="brand.300" fontWeight="600">
                {brightness === 100 ? '100% (chuẩn)' : `${brightness}%`}
              </Text>
            </Flex>
            <Slider value={brightness} onChange={setBrightness} min={1} max={100} step={1} size="sm">
              <SliderTrack><SliderFilledTrack bg="brand.500" /></SliderTrack>
              <SliderThumb boxSize={3} />
            </Slider>
            <Flex justify="space-between" mt={1}>
              <Text fontSize="9px" color="gray.600">Tối (1%)</Text>
              <Text fontSize="9px" color="gray.500">100% = chuẩn model</Text>
            </Flex>
          </Box>

          <Button
            colorScheme="brand" size="lg" w="full"
            onClick={() => handleRun()}
            isDisabled={loading || files.length === 0}
            isLoading={loading}
            loadingText="Đang xử lý..."
          >
            🚀 Ghép & Xử lý
          </Button>
        </VStack>
      </SimpleGrid>
    </Box>
  )
}

