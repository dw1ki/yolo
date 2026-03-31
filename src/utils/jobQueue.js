/**
 * Simple in-memory job queue untuk handling long-running tasks
 * Production: gunakan Redis + Bull untuk scalability
 */

class JobQueue {
  constructor() {
    this.jobs = new Map();
    this.jobCounter = 0;
  }

  /**
   * Create new job
   */
  createJob(taskName, data = {}) {
    this.jobCounter++;
    const jobId = `job_${Date.now()}_${this.jobCounter}`;
    
    const job = {
      id: jobId,
      taskName,
      status: 'pending', // pending, processing, completed, failed
      progress: 0,
      data,
      result: null,
      error: null,
      createdAt: new Date(),
      updatedAt: new Date(),
      startedAt: null,
      completedAt: null,
    };

    this.jobs.set(jobId, job);
    console.log(`✅ Job created: ${jobId} (${taskName})`);
    return jobId;
  }

  /**
   * Get job by ID
   */
  getJob(jobId) {
    return this.jobs.get(jobId) || null;
  }

  /**
   * Update job progress
   */
  updateProgress(jobId, progress, message = '') {
    const job = this.jobs.get(jobId);
    if (job) {
      job.progress = Math.min(progress, 100);
      job.updatedAt = new Date();
      if (message) job.message = message;
      console.log(`📊 [${jobId}] Progress: ${progress}% ${message}`);
    }
    return job;
  }

  /**
   * Mark job as processing
   */
  startJob(jobId) {
    const job = this.jobs.get(jobId);
    if (job) {
      job.status = 'processing';
      job.startedAt = new Date();
      job.updatedAt = new Date();
      console.log(`🔄 [${jobId}] Started processing`);
    }
    return job;
  }

  /**
   * Mark job as completed
   */
  completeJob(jobId, result) {
    const job = this.jobs.get(jobId);
    if (job) {
      job.status = 'completed';
      job.progress = 100;
      job.result = result;
      job.completedAt = new Date();
      job.updatedAt = new Date();
      console.log(`✅ [${jobId}] Completed`);
    }
    return job;
  }

  /**
   * Mark job as failed
   */
  failJob(jobId, error) {
    const job = this.jobs.get(jobId);
    if (job) {
      job.status = 'failed';
      job.error = error instanceof Error ? error.message : error;
      job.completedAt = new Date();
      job.updatedAt = new Date();
      console.log(`❌ [${jobId}] Failed: ${job.error}`);
    }
    return job;
  }

  /**
   * Delete old jobs (cleanup)
   */
  cleanup(maxAgeMinutes = 60) {
    const now = Date.now();
    const maxAge = maxAgeMinutes * 60 * 1000;

    for (const [jobId, job] of this.jobs.entries()) {
      if (now - job.updatedAt.getTime() > maxAge) {
        this.jobs.delete(jobId);
        console.log(`🗑️ Cleaned up job: ${jobId}`);
      }
    }
  }
}

// Singleton instance
export const jobQueue = new JobQueue();

// Cleanup old jobs every 10 minutes
setInterval(() => {
  jobQueue.cleanup(60);
}, 10 * 60 * 1000);

export default jobQueue;
