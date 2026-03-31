// Middleware untuk membatasi akses berdasarkan role
export const adminOnly = (req, res, next) => {
  if (req.user.role !== "admin") {
    return res.status(403).json({ message: "Access denied: Admin only" });
  }
  next();
};

export const surveyorOnly = (req, res, next) => {
  if (req.user.role !== "surveyor") {
    return res.status(403).json({ message: "Access denied: Surveyor only" });
  }
  next();
};

export const userOnly = (req, res, next) => {
  if (req.user.role !== "user") {
    return res.status(403).json({ message: "Access denied: User only" });
  }
  next();
};
