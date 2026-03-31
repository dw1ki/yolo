import Perhitungan from "../models/perhitungan.js";

export const getDashboard = async (req, res) => {
  try {
    const data = await Perhitungan.find();

    if (!data || data.length === 0) {
      return res.json({
        totalKendaraan: 0,
        avgDJ: 0,
        losCount: {},
        message: "No data available"
      });
    }

    const totalKendaraan = data.reduce((s, d) => s + (d.volume?.total || 0), 0);
    const avgDJ = (
      data.reduce((s, d) => s + (d.DJ || 0), 0) / data.length
    ).toFixed(3);

    const losCount = {};
    data.forEach(d => {
      if (d.LOS) {
        losCount[d.LOS] = (losCount[d.LOS] || 0) + 1;
      }
    });

    res.json({
      totalKendaraan,
      avgDJ,
      losCount,
      dataPoints: data.length
    });
  } catch (error) {
    console.error('Dashboard error:', error);
    res.status(500).json({
      error: 'Failed to fetch dashboard data',
      totalKendaraan: 0,
      avgDJ: 0,
      losCount: {}
    });
  }
};
