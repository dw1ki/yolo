import express from "express";
import {
  createUser,
  getAllUsers,
  getUserById,
  updateUserRole,
  deactivateUser,
  deleteUser,
  updateProfile,
} from "../controllers/userController.js";
import { protect, authorize } from "../middlewares/auth.js";

const router = express.Router();

// Protected routes - All authenticated users
router.get("/me", protect, (req, res) => {
  res.json({
    success: true,
    user: req.user,
  });
});

router.put("/profile", protect, updateProfile);

// Admin only routes
router.post("/", protect, authorize("admin"), createUser);
router.get("/", protect, authorize("admin"), getAllUsers);
router.get("/:id", protect, authorize("admin"), getUserById);
router.put("/:id/role", protect, authorize("admin"), updateUserRole);
router.put("/:id/deactivate", protect, authorize("admin"), deactivateUser);
router.delete("/:id", protect, authorize("admin"), deleteUser);

export default router;
