import React, { useState, useEffect } from 'react'
import {
  Modal, ModalOverlay, ModalContent, ModalCloseButton,
  ModalBody, Image, Box, Text, IconButton,
} from '@chakra-ui/react'

// Inline SVG chevrons — no extra icon package needed
const ChevronLeft = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="15 18 9 12 15 6" />
  </svg>
)
const ChevronRight = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6" />
  </svg>
)
const DownloadIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
)

function handleDownload(src, name = 'image.png') {
  const a = document.createElement('a')
  a.href = src
  a.download = name
  a.click()
}

/**
 * ZoomableImage — thumbnail that opens a lightbox on click.
 *
 * Props:
 *  src           — image URL
 *  alt           — alt text (optional)
 *  caption       — caption shown in lightbox (optional)
 *  downloadSrc   — if provided, show a download button (defaults to src)
 *  downloadName  — filename for download (default: image.png)
 *  gallery       — [{src, caption, downloadSrc}] for prev/next navigation (optional)
 *  galleryIndex  — index of this image inside gallery (optional)
 *  ...rest       — forwarded to <Image>
 */
export function ZoomableImage({ src, alt = '', caption, downloadSrc, downloadName = 'image.png', gallery, galleryIndex = 0, ...rest }) {
  const [open, setOpen] = useState(false)
  const images = gallery || [{ src, caption, downloadSrc: downloadSrc || src }]
  const dlSrc = downloadSrc || src

  return (
    <>
      <Box
        position="relative"
        cursor="zoom-in"
        borderRadius="md"
        overflow="hidden"
        display="flex"
        w="100%"
        onClick={() => setOpen(true)}
      >
        <Image src={src} alt={alt} w="100%" {...rest} />
      </Box>

      <LightboxModal
        isOpen={open}
        onClose={() => setOpen(false)}
        images={images}
        initialIndex={gallery ? galleryIndex : 0}
      />
    </>
  )
}

/**
 * LightboxModal — fullscreen viewer with optional prev/next navigation.
 *
 * Props:
 *  isOpen       — boolean
 *  onClose      — function
 *  images       — [{src, caption}]
 *  initialIndex — starting index (default 0)
 */
export function LightboxModal({ isOpen, onClose, images = [], initialIndex = 0 }) {
  const [idx, setIdx] = useState(initialIndex)

  // Reset to correct index whenever modal opens
  useEffect(() => {
    if (isOpen) setIdx(initialIndex)
  }, [isOpen, initialIndex])

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return
    const handleKey = (e) => {
      if (e.key === 'ArrowLeft') setIdx((i) => Math.max(0, i - 1))
      if (e.key === 'ArrowRight') setIdx((i) => Math.min(images.length - 1, i + 1))
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isOpen, images.length])

  const current = images[idx] || {}
  const hasPrev = idx > 0
  const hasNext = idx < images.length - 1

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="full" isCentered>
      <ModalOverlay bg="blackAlpha.900" backdropFilter="blur(4px)" />
      <ModalContent
        bg="transparent" boxShadow="none" maxW="100vw" maxH="100vh"
        display="flex" alignItems="center" justifyContent="center"
        onClick={onClose}
      >
        {/* Close button */}
        <ModalCloseButton
          color="white" size="lg" top={4} right={4}
          bg="whiteAlpha.200" _hover={{ bg: 'whiteAlpha.400' }}
          onClick={(e) => { e.stopPropagation(); onClose() }}
          zIndex={10}
        />

        {/* Download button top-left */}
        {current.src && (
          <IconButton
            aria-label="Download"
            icon={<DownloadIcon />}
            position="fixed" left={4} top={4}
            bg="whiteAlpha.200" color="white"
            _hover={{ bg: 'whiteAlpha.400' }}
            size="md" borderRadius="md" zIndex={10}
            title="Tải xuống"
            onClick={(e) => { e.stopPropagation(); handleDownload(current.downloadSrc || current.src, 'image.png') }}
          />
        )}

        {/* Prev button */}
        {hasPrev && (
          <IconButton
            aria-label="Previous"
            icon={<ChevronLeft />}
            position="fixed" left={4} top="50%" transform="translateY(-50%)"
            bg="whiteAlpha.200" color="white"
            _hover={{ bg: 'whiteAlpha.400' }}
            size="lg" borderRadius="full" zIndex={10}
            onClick={(e) => { e.stopPropagation(); setIdx(idx - 1) }}
          />
        )}

        {/* Next button */}
        {hasNext && (
          <IconButton
            aria-label="Next"
            icon={<ChevronRight />}
            position="fixed" right={4} top="50%" transform="translateY(-50%)"
            bg="whiteAlpha.200" color="white"
            _hover={{ bg: 'whiteAlpha.400' }}
            size="lg" borderRadius="full" zIndex={10}
            onClick={(e) => { e.stopPropagation(); setIdx(idx + 1) }}
          />
        )}

        <ModalBody
          display="flex" flexDir="column" alignItems="center" justifyContent="center"
          p={4} onClick={(e) => e.stopPropagation()}
        >
          <Image
            src={current.src}
            alt={current.caption || ''}
            maxW="90vw"
            maxH="85vh"
            objectFit="contain"
            borderRadius="lg"
            boxShadow="0 8px 64px rgba(0,0,0,0.8)"
          />
          {current.caption && (
            <Text color="whiteAlpha.800" fontSize="sm" mt={3} textAlign="center">
              {current.caption}
            </Text>
          )}
          {images.length > 1 ? (
            <Text color="whiteAlpha.500" fontSize="xs" mt={2}>
              {idx + 1} / {images.length} · ← → để điều hướng · Esc để đóng
            </Text>
          ) : (
            <Text color="whiteAlpha.500" fontSize="xs" mt={2}>
              Click ngoài ảnh hoặc nhấn Esc để đóng
            </Text>
          )}
        </ModalBody>
      </ModalContent>
    </Modal>
  )
}

export default ZoomableImage
