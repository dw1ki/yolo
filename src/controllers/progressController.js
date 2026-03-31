import { progressStore } from "../store/progressStore.js"

export const detectProgress = (req, res) => {
  const { jobId } = req.params

  res.setHeader("Content-Type", "text/event-stream")
  res.setHeader("Cache-Control", "no-cache")
  res.setHeader("Connection", "keep-alive")

  const interval = setInterval(() => {
    const progress = progressStore.get(jobId)

    if (!progress) return

    res.write(`data: ${JSON.stringify(progress)}\n\n`)

    if (progress.progress >= 100 || progress.progress === -1) {
      clearInterval(interval)
      res.end()
    }
  }, 500)
}
