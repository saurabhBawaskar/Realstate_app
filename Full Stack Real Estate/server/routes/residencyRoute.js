import express from "express";
import { createResidency, getAllResidencies, getResidency } from "../controllers/resdCntrl.js";
const router = express.Router();

router.post("/create", jwtCheck, createResidency)
router.get("/allresd", getAllResidencies)
router.get("/:id", getResidency)
export {router as residencyRoute}
