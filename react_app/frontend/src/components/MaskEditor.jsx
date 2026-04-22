/**
 * MaskEditor — Canvas-based mask editor with multiple drawing tools.
 *
 * Tools: Brush, Rectangle, Circle, Polygon, Eraser
 * Features: Undo/Redo, Scroll-to-zoom, Pan (space+drag / middle-click),
 *           Red overlay at 60% opacity, Lightbox when no tool selected.
 *
 * Props:
 *   imageUrl    — URL of the loaded image
 *   height      — total height in px (default 500)
 *   onOpenLightbox — called when user clicks canvas with no tool selected
 *
 * Ref API:
 *   getMaskBlob() → Promise<Blob>   — returns the black/white mask as PNG
 *   clearMask()                      — resets mask to empty
 */
import React, {
    useState, useRef, useEffect, useCallback,
    forwardRef, useImperativeHandle,
} from 'react'
import {
    Box, HStack, IconButton, Tooltip, Flex, Text, Divider,
    Slider, SliderTrack, SliderFilledTrack, SliderThumb,
} from '@chakra-ui/react'

/* ── Inline SVG Icons ── */
const IconBrush = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M18.37 2.63 14 7l-1.59-1.59a2 2 0 0 0-2.82 0L8 7l9 9 1.59-1.59a2 2 0 0 0 0-2.82L17 10l4.37-4.37a2.12 2.12 0 1 0-3-3Z" />
        <path d="M9 8c-2 3-4 3.5-7 4l8 10c2-1 6-5 6-7" />
        <path d="M14.5 17.5 4.5 15" />
    </svg>
)
const IconRect = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" />
    </svg>
)
const IconCircle = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
    </svg>
)
const IconPolygon = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2l8 5v6l-8 5-8-5V7z" />
    </svg>
)
const IconEraser = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21" />
        <path d="M22 21H7" />
        <path d="m5 11 9 9" />
    </svg>
)
const IconUndo = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 7v6h6" />
        <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
    </svg>
)
const IconRedo = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 7v6h-6" />
        <path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13" />
    </svg>
)
const IconTrash = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 6h18" />
        <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
        <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
)
const IconZoomReset = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
        <line x1="8" y1="11" x2="14" y2="11" />
    </svg>
)

const MASK_OVERLAY = 'rgba(255, 0, 0, 0.4)'
const MASK_STROKE = '#ff0000'
const TOOLBAR_H = 36

const MaskEditor = forwardRef(function MaskEditor(
    { imageUrl, height = 500, onOpenLightbox },
    ref,
) {
    const containerRef = useRef(null)
    const canvasRef = useRef(null)
    const maskRef = useRef(null)   // hidden black/white full-res mask
    const imgElRef = useRef(null)   // HTMLImageElement
    const dimsRef = useRef(null)   // { natW, natH, dispW, dispH }

    const [tool, setTool] = useState('brush')
    const [brushSize, setBrushSize] = useState(30)
    const zoomRef = useRef(1)
    const panRef = useRef({ x: 0, y: 0 })
    const [, bump] = useState(0)
    const forceRender = useCallback(() => bump((n) => n + 1), [])

    // History
    const historyRef = useRef([])
    const hIdxRef = useRef(-1)
    const [canUndo, setCanUndo] = useState(false)
    const [canRedo, setCanRedo] = useState(false)

    // Drawing state (refs — no re-render during drag)
    const drawingRef = useRef(false)
    const lastPosRef = useRef(null)
    const shapeStartRef = useRef(null)
    const polyPointsRef = useRef([])
    const baseDisplayRef = useRef(null)   // ImageData snapshot for shape preview
    const isPanningRef = useRef(false)
    const panStartRef = useRef(null)
    const spaceHeldRef = useRef(false)
    const toolRef = useRef(tool)
    const brushSizeRef = useRef(brushSize)

    // Keep refs in sync with state
    useEffect(() => { toolRef.current = tool }, [tool])
    useEffect(() => { brushSizeRef.current = brushSize }, [brushSize])

    /* ── Expose to parent ── */
    useImperativeHandle(ref, () => ({
        getMaskBlob: () =>
            new Promise((resolve) => {
                if (!maskRef.current) return resolve(null)
                maskRef.current.toBlob(resolve, 'image/png')
            }),
        clearMask: clearMaskFn,
        /** Load an external mask image (white = masked area) onto the mask canvas */
        loadMask: (maskUrl) => new Promise((resolve) => {
            const mask = maskRef.current
            const d = dimsRef.current
            if (!mask || !d) return resolve(false)
            const img = new Image()
            img.crossOrigin = 'anonymous'
            img.onload = () => {
                const mctx = mask.getContext('2d')
                mctx.fillStyle = 'black'
                mctx.fillRect(0, 0, d.natW, d.natH)
                mctx.drawImage(img, 0, 0, d.natW, d.natH)
                redraw()
                saveHistory()
                resolve(true)
            }
            img.onerror = () => resolve(false)
            img.src = maskUrl
        }),
    }))

    /* ── Coordinate helpers ── */
    const getCanvasPos = useCallback((e) => {
        const canvas = canvasRef.current
        if (!canvas) return { x: 0, y: 0 }
        const rect = canvas.getBoundingClientRect()
        return {
            x: ((e.clientX - rect.left) / rect.width) * canvas.width,
            y: ((e.clientY - rect.top) / rect.height) * canvas.height,
        }
    }, [])

    const toMask = useCallback((cx, cy) => {
        const d = dimsRef.current
        if (!d) return { x: 0, y: 0 }
        return { x: (cx / d.dispW) * d.natW, y: (cy / d.dispH) * d.natH }
    }, [])

    /* ── Redraw display canvas (image + red overlay) ── */
    const redraw = useCallback(() => {
        const canvas = canvasRef.current
        const img = imgElRef.current
        const mask = maskRef.current
        const d = dimsRef.current
        if (!canvas || !img || !mask || !d) return

        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, d.dispW, d.dispH)
        ctx.drawImage(img, 0, 0, d.dispW, d.dispH)

        // Draw mask scaled to display, recolor to red 60%
        const tmp = document.createElement('canvas')
        tmp.width = d.dispW
        tmp.height = d.dispH
        const tctx = tmp.getContext('2d')
        tctx.drawImage(mask, 0, 0, d.dispW, d.dispH)
        const od = tctx.getImageData(0, 0, d.dispW, d.dispH)
        const px = od.data
        for (let i = 0; i < px.length; i += 4) {
            if (px[i] > 127 || px[i + 1] > 127 || px[i + 2] > 127) {
                px[i] = 255; px[i + 1] = 0; px[i + 2] = 0; px[i + 3] = 153
            } else {
                px[i + 3] = 0
            }
        }
        tctx.putImageData(od, 0, 0)
        ctx.drawImage(tmp, 0, 0)
    }, [])

    /* ── History ── */
    const saveHistory = useCallback(() => {
        const mask = maskRef.current
        const d = dimsRef.current
        if (!mask || !d) return
        const data = mask.getContext('2d').getImageData(0, 0, d.natW, d.natH)
        historyRef.current = historyRef.current.slice(0, hIdxRef.current + 1)
        historyRef.current.push(data)
        hIdxRef.current = historyRef.current.length - 1
        if (historyRef.current.length > 40) {
            historyRef.current.shift()
            hIdxRef.current--
        }
        setCanUndo(hIdxRef.current > 0)
        setCanRedo(false)
    }, [])

    const restoreHistory = useCallback((idx) => {
        const mask = maskRef.current
        if (!mask || !historyRef.current[idx]) return
        mask.getContext('2d').putImageData(historyRef.current[idx], 0, 0)
        redraw()
        setCanUndo(idx > 0)
        setCanRedo(idx < historyRef.current.length - 1)
    }, [redraw])

    const undo = useCallback(() => {
        if (hIdxRef.current <= 0) return
        hIdxRef.current--
        restoreHistory(hIdxRef.current)
    }, [restoreHistory])

    const redo = useCallback(() => {
        if (hIdxRef.current >= historyRef.current.length - 1) return
        hIdxRef.current++
        restoreHistory(hIdxRef.current)
    }, [restoreHistory])

    /* ── Clear mask ── */
    // eslint-disable-next-line react-hooks/exhaustive-deps
    const clearMaskFn = useCallback(() => {
        const mask = maskRef.current
        const d = dimsRef.current
        if (!mask || !d) return
        const mctx = mask.getContext('2d')
        mctx.fillStyle = 'black'
        mctx.fillRect(0, 0, d.natW, d.natH)
        polyPointsRef.current = []
        redraw()
        saveHistory()
    }, [redraw, saveHistory])

    /* ── Load image ── */
    useEffect(() => {
        if (!imageUrl) return
        const img = new Image()
        img.onload = () => {
            const ctr = containerRef.current
            if (!ctr) return
            const maxW = ctr.clientWidth - 16
            const maxH = height - TOOLBAR_H - 16
            const scale = Math.min(maxW / img.width, maxH / img.height, 1)
            const dispW = Math.round(img.width * scale)
            const dispH = Math.round(img.height * scale)
            dimsRef.current = { natW: img.width, natH: img.height, dispW, dispH }
            imgElRef.current = img

            const canvas = canvasRef.current
            canvas.width = dispW
            canvas.height = dispH

            const mask = document.createElement('canvas')
            mask.width = img.width
            mask.height = img.height
            const mctx = mask.getContext('2d')
            mctx.fillStyle = 'black'
            mctx.fillRect(0, 0, img.width, img.height)
            maskRef.current = mask

            zoomRef.current = 1
            panRef.current = { x: 0, y: 0 }
            polyPointsRef.current = []
            historyRef.current = []
            hIdxRef.current = -1

            // Draw image and init history
            const ctx = canvas.getContext('2d')
            ctx.drawImage(img, 0, 0, dispW, dispH)
            saveHistory()
            forceRender()
        }
        img.src = imageUrl
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imageUrl])

    /* ── Mask drawing primitives ── */
    const drawCircleOnMask = useCallback((cx, cy, size, erase) => {
        const mask = maskRef.current
        if (!mask) return
        const d = dimsRef.current
        if (!d) return
        const mctx = mask.getContext('2d')
        const mx = (cx / d.dispW) * d.natW
        const my = (cy / d.dispH) * d.natH
        const mr = (size / 2) * (d.natW / d.dispW)
        mctx.fillStyle = erase ? 'black' : 'white'
        mctx.beginPath()
        mctx.arc(mx, my, mr, 0, Math.PI * 2)
        mctx.fill()
    }, [])

    const drawLineOnMask = useCallback((x1, y1, x2, y2, size, erase) => {
        const dx = x2 - x1, dy = y2 - y1
        const dist = Math.sqrt(dx * dx + dy * dy)
        const steps = Math.max(Math.ceil(dist / (size / 4)), 1)
        for (let i = 0; i <= steps; i++) {
            const t = i / steps
            drawCircleOnMask(x1 + dx * t, y1 + dy * t, size, erase)
        }
    }, [drawCircleOnMask])

    const fillRectOnMask = useCallback((x1, y1, x2, y2) => {
        const mask = maskRef.current
        const d = dimsRef.current
        if (!mask || !d) return
        const mctx = mask.getContext('2d')
        const sX = d.natW / d.dispW, sY = d.natH / d.dispH
        const lx = Math.min(x1, x2) * sX, ly = Math.min(y1, y2) * sY
        const w = Math.abs(x2 - x1) * sX, h = Math.abs(y2 - y1) * sY
        mctx.fillStyle = 'white'
        mctx.fillRect(lx, ly, w, h)
    }, [])

    const fillEllipseOnMask = useCallback((x1, y1, x2, y2) => {
        const mask = maskRef.current
        const d = dimsRef.current
        if (!mask || !d) return
        const mctx = mask.getContext('2d')
        const sX = d.natW / d.dispW, sY = d.natH / d.dispH
        const cx = ((x1 + x2) / 2) * sX, cy = ((y1 + y2) / 2) * sY
        const rx = (Math.abs(x2 - x1) / 2) * sX, ry = (Math.abs(y2 - y1) / 2) * sY
        mctx.fillStyle = 'white'
        mctx.beginPath()
        mctx.ellipse(cx, cy, Math.max(rx, 1), Math.max(ry, 1), 0, 0, Math.PI * 2)
        mctx.fill()
    }, [])

    const fillPolygonOnMask = useCallback((points) => {
        if (points.length < 3) return
        const mask = maskRef.current
        const d = dimsRef.current
        if (!mask || !d) return
        const mctx = mask.getContext('2d')
        const sX = d.natW / d.dispW, sY = d.natH / d.dispH
        mctx.fillStyle = 'white'
        mctx.beginPath()
        mctx.moveTo(points[0].x * sX, points[0].y * sY)
        for (let i = 1; i < points.length; i++) {
            mctx.lineTo(points[i].x * sX, points[i].y * sY)
        }
        mctx.closePath()
        mctx.fill()
    }, [])

    /* ── Shape preview on display canvas ── */
    const drawShapePreview = useCallback((start, cur, type) => {
        const canvas = canvasRef.current
        if (!canvas || !baseDisplayRef.current) return
        const ctx = canvas.getContext('2d')
        ctx.putImageData(baseDisplayRef.current, 0, 0)
        ctx.fillStyle = MASK_OVERLAY
        ctx.strokeStyle = MASK_STROKE
        ctx.lineWidth = 2
        if (type === 'rect') {
            const x = Math.min(start.x, cur.x), y = Math.min(start.y, cur.y)
            const w = Math.abs(cur.x - start.x), h = Math.abs(cur.y - start.y)
            ctx.fillRect(x, y, w, h)
            ctx.strokeRect(x, y, w, h)
        } else if (type === 'circle') {
            const cx = (start.x + cur.x) / 2, cy = (start.y + cur.y) / 2
            const rx = Math.abs(cur.x - start.x) / 2, ry = Math.abs(cur.y - start.y) / 2
            ctx.beginPath()
            ctx.ellipse(cx, cy, Math.max(rx, 1), Math.max(ry, 1), 0, 0, Math.PI * 2)
            ctx.fill()
            ctx.stroke()
        }
    }, [])

    const drawPolygonPreview = useCallback((points, cursorPos) => {
        const canvas = canvasRef.current
        if (!canvas || !baseDisplayRef.current) return
        const ctx = canvas.getContext('2d')
        ctx.putImageData(baseDisplayRef.current, 0, 0)
        if (!points.length) return

        // Determine the last point (cursor or last vertex)
        const lastPt = cursorPos || points[points.length - 1]

        // Check if cursor is near the first point (close zone)
        const nearFirst = cursorPos && points.length >= 3 &&
            Math.hypot(cursorPos.x - points[0].x, cursorPos.y - points[0].y) < 12

        // Draw filled polygon preview only when closing
        if (nearFirst) {
            ctx.fillStyle = MASK_OVERLAY
            ctx.beginPath()
            ctx.moveTo(points[0].x, points[0].y)
            for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y)
            ctx.lineTo(cursorPos.x, cursorPos.y)
            ctx.closePath()
            ctx.fill()
        }

        // Draw solid lines along placed vertices
        ctx.strokeStyle = MASK_STROKE
        ctx.lineWidth = 2
        ctx.setLineDash([])
        ctx.beginPath()
        ctx.moveTo(points[0].x, points[0].y)
        for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y)
        ctx.stroke()

        // Draw dashed line from last vertex to cursor
        if (cursorPos) {
            ctx.setLineDash([6, 4])
            ctx.beginPath()
            ctx.moveTo(points[points.length - 1].x, points[points.length - 1].y)
            ctx.lineTo(cursorPos.x, cursorPos.y)
            ctx.stroke()
            ctx.setLineDash([])
        }

        // Vertex dots
        ctx.fillStyle = MASK_STROKE
        for (const p of points) {
            ctx.beginPath()
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2)
            ctx.fill()
        }
        // Highlight first point when near (close indicator)
        if (nearFirst) {
            ctx.strokeStyle = '#ffffff'
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.arc(points[0].x, points[0].y, 8, 0, Math.PI * 2)
            ctx.strokeePath()
            ctx.fill()
        }

        // Draw solid lines along placed vertices
        ctx.strokeStyle = MASK_STROKE
        ctx.lineWidth = 2
        ctx.setLineDash([])
        ctx.beginPath()
        ctx.moveTo(points[0].x, points[0].y)
        for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y)
        ctx.stroke()

        // Draw dashed line from last vertex to cursor
        if (cursorPos) {
            ctx.setLineDash([6, 4])
            ctx.beginPath()
            ctx.moveTo(points[points.length - 1].x, points[points.length - 1].y)
            ctx.lineTo(cursorPos.x, cursorPos.y)
            ctx.stroke()
            ctx.setLineDash([])
        }

        // Vertex dots
        ctx.fillStyle = MASK_STROKE
        for (const p of points) {
            ctx.beginPath()
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2)
            ctx.fill()
        }
        // Highlight first point when near (close indicator)
        if (nearFirst) {
            ctx.strokeStyle = '#ffffff'
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.arc(points[0].x, points[0].y, 8, 0, Math.PI * 2)
            ctx.stroke()
        }
    }, [])

    /* ── Mouse handlers ── */
    const handleMouseDown = useCallback((e) => {
        e.preventDefault()
        const t = toolRef.current
        const bs = brushSizeRef.current

        // Middle-mouse or space = pan
        if (e.button === 1 || spaceHeldRef.current) {
            isPanningRef.current = true
            panStartRef.current = {
                x: e.clientX - panRef.current.x,
                y: e.clientY - panRef.current.y,
            }
            return
        }
        if (e.button !== 0) return

        // No tool → lightbox
        if (!t) { onOpenLightbox?.(); return }

        const pos = getCanvasPos(e)

        if (t === 'polygon') {
            if (polyPointsRef.current.length === 0) {
                baseDisplayRef.current = canvasRef.current.getContext('2d')
                    .getImageData(0, 0, canvasRef.current.width, canvasRef.current.height)
            }
            // Close polygon if clicking near first point
            if (polyPointsRef.current.length >= 3) {
                const f = polyPointsRef.current[0]
                if (Math.hypot(pos.x - f.x, pos.y - f.y) < 12) {
                    fillPolygonOnMask(polyPointsRef.current)
                    polyPointsRef.current = []
                    redraw()
                    saveHistory()
                    return
                }
            }
            polyPointsRef.current.push(pos)
            drawPolygonPreview(polyPointsRef.current, null)
            return
        }

        if (t === 'brush' || t === 'eraser') {
            drawingRef.current = true
            lastPosRef.current = pos
            drawCircleOnMask(pos.x, pos.y, bs, t === 'eraser')
            redraw()
        } else if (t === 'rect' || t === 'circle') {
            shapeStartRef.current = pos
            drawingRef.current = true
            baseDisplayRef.current = canvasRef.current.getContext('2d')
                .getImageData(0, 0, canvasRef.current.width, canvasRef.current.height)
        }
    }, [getCanvasPos, drawCircleOnMask, redraw, saveHistory,
        fillPolygonOnMask, drawPolygonPreview, onOpenLightbox])

    const handleMouseMove = useCallback((e) => {
        if (isPanningRef.current && panStartRef.current) {
            panRef.current = {
                x: e.clientX - panStartRef.current.x,
                y: e.clientY - panStartRef.current.y,
            }
            forceRender()
            return
        }

        const t = toolRef.current
        const bs = brushSizeRef.current

        if (t === 'polygon' && polyPointsRef.current.length > 0) {
            drawPolygonPreview(polyPointsRef.current, getCanvasPos(e))
            return
        }

        if (!drawingRef.current) return
        const pos = getCanvasPos(e)

        if (t === 'brush' || t === 'eraser') {
            if (lastPosRef.current) {
                drawLineOnMask(lastPosRef.current.x, lastPosRef.current.y,
                    pos.x, pos.y, bs, t === 'eraser')
            }
            lastPosRef.current = pos
            redraw()
        } else if ((t === 'rect' || t === 'circle') && shapeStartRef.current) {
            drawShapePreview(shapeStartRef.current, pos, t)
        }
    }, [getCanvasPos, drawLineOnMask, redraw, drawShapePreview, drawPolygonPreview, forceRender])

    const handleMouseUp = useCallback((e) => {
        if (isPanningRef.current) {
            isPanningRef.current = false
            panStartRef.current = null
            return
        }
        if (!drawingRef.current) return
        drawingRef.current = false

        const t = toolRef.current
        const pos = getCanvasPos(e)

        if (t === 'rect' && shapeStartRef.current) {
            fillRectOnMask(shapeStartRef.current.x, shapeStartRef.current.y, pos.x, pos.y)
            shapeStartRef.current = null
            redraw(); saveHistory()
        } else if (t === 'circle' && shapeStartRef.current) {
            fillEllipseOnMask(shapeStartRef.current.x, shapeStartRef.current.y, pos.x, pos.y)
            shapeStartRef.current = null
            redraw(); saveHistory()
        } else if (t === 'brush' || t === 'eraser') {
            lastPosRef.current = null
            saveHistory()
        }
    }, [getCanvasPos, fillRectOnMask, fillEllipseOnMask, redraw, saveHistory])

    const handleDoubleClick = useCallback(() => {
        if (toolRef.current === 'polygon' && polyPointsRef.current.length >= 3) {
            fillPolygonOnMask(polyPointsRef.current)
            polyPointsRef.current = []
            redraw(); saveHistory()
        }
    }, [fillPolygonOnMask, redraw, saveHistory])

    /* ── Wheel zoom ── */
    const handleWheel = useCallback((e) => {
        e.preventDefault()
        const factor = e.deltaY > 0 ? 0.9 : 1.1
        zoomRef.current = Math.min(Math.max(zoomRef.current * factor, 0.5), 5)
        forceRender()
    }, [forceRender])

    const resetZoom = useCallback(() => {
        zoomRef.current = 1
        panRef.current = { x: 0, y: 0 }
        forceRender()
    }, [forceRender])

    /* ── Keyboard shortcuts ── */
    useEffect(() => {
        const onKeyDown = (e) => {
            if (e.code === 'Space') { e.preventDefault(); spaceHeldRef.current = true }
            if (e.code === 'Escape' && polyPointsRef.current.length > 0) {
                polyPointsRef.current = []
                redraw()
            }
            if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ') {
                e.preventDefault()
                if (e.shiftKey) redo(); else undo()
            }
        }
        const onKeyUp = (e) => {
            if (e.code === 'Space') spaceHeldRef.current = false
        }
        window.addEventListener('keydown', onKeyDown)
        window.addEventListener('keyup', onKeyUp)
        return () => {
            window.removeEventListener('keydown', onKeyDown)
            window.removeEventListener('keyup', onKeyUp)
        }
    }, [redraw, undo, redo])

    /* ── Cursor ── */
    const cursor = !tool ? 'pointer'
        : spaceHeldRef.current ? 'grab'
            : 'crosshair'

    const canvasH = height - TOOLBAR_H - 12

    /* ── Render ── */
    return (
        <Box ref={containerRef} h={`${height}px`} position="relative" userSelect="none">
            {/* Toolbar */}
            <Flex
                gap={1} p="4px 6px" bg="whiteAlpha.50" borderRadius="md"
                align="center" flexWrap="wrap" mb="6px"
                border="1px solid" borderColor="whiteAlpha.100"
            >
                {[
                    { id: 'brush', icon: <IconBrush />, label: 'Brush (vẽ tự do)' },
                    { id: 'rect', icon: <IconRect />, label: 'Hình chữ nhật' },
                    { id: 'circle', icon: <IconCircle />, label: 'Hình tròn' },
                    { id: 'polygon', icon: <IconPolygon />, label: 'Polygon (nối điểm, double-click để đóng)' },
                ].map((t) => (
                    <Tooltip key={t.id} label={t.label} fontSize="xs" hasArrow>
                        <IconButton
                            size="xs"
                            variant={tool === t.id ? 'solid' : 'ghost'}
                            colorScheme={tool === t.id ? 'brand' : 'gray'}
                            icon={t.icon}
                            onClick={() => {
                                polyPointsRef.current = []
                                setTool(tool === t.id ? null : t.id)
                            }}
                            aria-label={t.label}
                        />
                    </Tooltip>
                ))}

                <Divider orientation="vertical" h="18px" mx={1} borderColor="whiteAlpha.200" />

                <Tooltip label="Eraser (xóa vùng mask)" fontSize="xs" hasArrow>
                    <IconButton
                        size="xs"
                        variant={tool === 'eraser' ? 'solid' : 'ghost'}
                        colorScheme={tool === 'eraser' ? 'red' : 'gray'}
                        icon={<IconEraser />}
                        onClick={() => {
                            polyPointsRef.current = []
                            setTool(tool === 'eraser' ? null : 'eraser')
                        }}
                        aria-label="Eraser"
                    />
                </Tooltip>

                <Divider orientation="vertical" h="18px" mx={1} borderColor="whiteAlpha.200" />

                <Tooltip label="Undo (Ctrl+Z)" fontSize="xs" hasArrow>
                    <IconButton
                        size="xs" variant="ghost" icon={<IconUndo />}
                        onClick={undo} isDisabled={!canUndo} aria-label="Undo"
                    />
                </Tooltip>
                <Tooltip label="Redo (Ctrl+Shift+Z)" fontSize="xs" hasArrow>
                    <IconButton
                        size="xs" variant="ghost" icon={<IconRedo />}
                        onClick={redo} isDisabled={!canRedo} aria-label="Redo"
                    />
                </Tooltip>

                <Divider orientation="vertical" h="18px" mx={1} borderColor="whiteAlpha.200" />

                <Tooltip label="Xóa toàn bộ mask" fontSize="xs" hasArrow>
                    <IconButton
                        size="xs" variant="ghost" colorScheme="red"
                        icon={<IconTrash />} onClick={clearMaskFn}
                        aria-label="Clear all"
                    />
                </Tooltip>

                {zoomRef.current !== 1 && (
                    <>
                        <Divider orientation="vertical" h="18px" mx={1} borderColor="whiteAlpha.200" />
                        <Tooltip label="Reset zoom" fontSize="xs" hasArrow>
                            <IconButton
                                size="xs" variant="ghost" icon={<IconZoomReset />}
                                onClick={resetZoom} aria-label="Reset zoom"
                            />
                        </Tooltip>
                        <Text fontSize="9px" color="gray.500">{Math.round(zoomRef.current * 100)}%</Text>
                    </>
                )}

                {(tool === 'brush' || tool === 'eraser') && (
                    <Flex align="center" gap={1} ml="auto">
                        <Text fontSize="xs" color="gray.400" whiteSpace="nowrap">Size</Text>
                        <Slider
                            value={brushSize} onChange={setBrushSize}
                            min={3} max={120} w="70px" size="sm"
                        >
                            <SliderTrack><SliderFilledTrack bg="brand.500" /></SliderTrack>
                            <SliderThumb boxSize={2} />
                        </Slider>
                        <Text fontSize="xs" color="brand.300" fontWeight="600" w="28px" textAlign="right">
                            {brushSize}
                        </Text>
                    </Flex>
                )}
            </Flex>

            {/* Canvas area */}
            <Box
                overflow="hidden"
                h={`${canvasH}px`}
                display="flex" alignItems="center" justifyContent="center"
                borderRadius="md"
                onWheel={handleWheel}
            >
                <canvas
                    ref={canvasRef}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={(e) => { if (drawingRef.current) handleMouseUp(e) }}
                    onDoubleClick={handleDoubleClick}
                    onContextMenu={(e) => e.preventDefault()}
                    style={{
                        maxWidth: '100%',
                        maxHeight: '100%',
                        cursor,
                        display: 'block',
                        borderRadius: '6px',
                        transform: `translate(${panRef.current.x}px, ${panRef.current.y}px) scale(${zoomRef.current})`,
                        transformOrigin: 'center center',
                        willChange: 'transform',
                    }}
                />
            </Box>
        </Box>
    )
})

export default MaskEditor
