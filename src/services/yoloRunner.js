import { progressStore } from "../store/progressStore.js"

export const runYOLO = async (jobId) => {
  const totalFrames = 4500

  for (let i = 1; i <= totalFrames; i++) {
    await new Promise((r) => setTimeout(r, 2))

    const percent = Math.round((i / totalFrames) * 100)

    progressStore.set(jobId, {
      progress: percent,
      message: `Processing frame ${i} / ${totalFrames}`
    })
  }

  progressStore.set(jobId, {
    progress: 100,
    message: "YOLO processing completed"
  })

  return {
    frames_processed: totalFrames,
    result: {
      kiri: { total: 45, mobil: 42, bus: 3, truk: 0 },
      kanan: { total: 36, mobil: 30, bus: 6, truk: 0 },
      total: 81
    }
  }
}
