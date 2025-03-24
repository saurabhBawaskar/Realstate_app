export const errorHandler = (err, req, res, next) => {
    console.error("Error details:", err);
    
    if (err.name === 'UnauthorizedError' || err.name === 'InvalidTokenError') {
        return res.status(401).json({ message: 'Invalid token', details: err.message });
    }

    res.status(500).json({ message: 'Something went wrong', details: err.message });
};