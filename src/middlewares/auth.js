import jwt from "jsonwebtoken";
import User from "../models/User.js";

// ================== PROTECT ROUTE ==================
// Verifikasi JWT token dan load user
export const protect = async (req, res, next) => {
  let token;

  // Get token dari header
  if (
    req.headers.authorization &&
    req.headers.authorization.startsWith("Bearer")
  ) {
    token = req.headers.authorization.split(" ")[1];
  }

  if (!token) {
    console.log("[Auth] No token in request to", req.path);
    return res.status(401).json({
      success: false,
      message: "Not authorized, token missing",
    });
  }

  try {
    console.log("[Auth] Verifying token for", req.path);
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = await User.findById(decoded.id);

    if (!req.user) {
      console.log("[Auth] User not found for token");
      return res.status(401).json({
        success: false,
        message: "User tidak ditemukan",
      });
    }

    if (!req.user.isActive) {
      console.log("[Auth] User account is inactive:", req.user.email);
      return res.status(401).json({
        success: false,
        message: "Akun Anda telah dinonaktifkan",
      });
    }

    console.log("[Auth] Token verified for user:", req.user.email);
    next();
  } catch (error) {
    console.error("[Auth] Token verification failed:", error.message);
    return res.status(401).json({
      success: false,
      message: "Not authorized, token failed",
      error: error.message,
    });
  }
};

// ================== ROLE-BASED ACCESS CONTROL ==================
// Check apakah user memiliki role yang diizinkan
export const authorize = (...allowedRoles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: "Not authenticated",
      });
    }

    if (!allowedRoles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        message: "Not authorized to access this resource",
      });
    }

    next();
  };
};

// ================== ERROR HANDLER ==================
export const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};
