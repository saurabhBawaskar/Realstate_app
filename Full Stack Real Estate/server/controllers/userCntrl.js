import asyncHandler from "express-async-handler";

export const createUser = asyncHandler(async (req, res) => {
  console.log("creating a user");
app.use(cookieParser());
//crete jwt secret key 
const jwtSecret = 'E3P5S8X4G2B7F1Y9D6I0C3R6K9T2Z1A7L';
//genearting salt using gensaltsync method of 10 rounds of processing ,algo will go through 2^10  1024 ronds of processing of hashing  
const bcryptSalt = bcrypt.genSaltSync(10);

app.get('/api/test', (req, res) => {

  res.json('Test Ok');
});

//apis for create and login
app.post('/api/register', async (req, res) => {
  const { username, email, password } = req.body;
  try {
    const userData = await User.create({
      username,
      email,
      password: bcrypt.hashSync(password, bcryptSalt),
    });
    res.json(userData);
  } catch (e) {
    console.log('Failed to create User', e);
    res.status(422).json('Failed to create User');
  }
});

//to retreive data from token we used this method in evry api
const getUserDataFromToken = req => {
  return new Promise((resolve, reject) => {
    jwt.verify(req.cookies.token, jwtSecret, {}, (e, userData) => {
      if (e) throw e;
      resolve(userData);
    });
  });
};

app.post('/api/login', async (req, res) => {
  mongoose.connect(process.env.MONGO_URL);
  const { username, password } = req.body;
  const userData = await User.findOne({ username });
  if (userData) {
    const passOk = bcrypt.compareSync(password, userData.password);
    if (passOk) {
      jwt.sign(
        {
          username: userData.username,
          id: userData._id,
        },
        jwtSecret,
        {},
        (e, token) => {
          if (e) throw e;
          //here cookie response will send to user computer and it will be stored there whic will contain token and userdata
          res.cookie('token', token).json(userData);
        },
      );
    } else {
      res.status(422).json('Password Did not match');
    }
  } else {
    res.json('User Details Not Found');
  }
});
app.get('/api/profile', (req, res) => {
  mongoose.connect(process.env.MONGO_URL);
  const { token } = req.cookies;
  // res.json({token})
  if (token) {
    // try and verify the token
    jwt.verify(token, jwtSecret, {}, async (err, user) => {
      if (err) throw err;
      const { username, email, _id } = await User.findById(user.id);
      res.json({ username, email, _id });
    });
  } else {
    res.json(null);
  }
});
app.post('/api/logout', (req, res) => {
  mongoose.connect(process.env.MONGO_URL);
  res.cookie('token', '').json(true);
});
// Post details to the database
app.post('/api/personal', async (req, res) => {
  mongoose.connect(process.env.MONGO_URL);
  // get token to verify the user
  const { token } = req.cookies;
  const { name, email, address, phone, website, linked } = req.body;
  jwt.verify(token, jwtSecret, {}, async (e, user) => {
    if (e) throw e;
    try {
      const postData = await Personal.create({
        user: user.id,
        name,
        email
      });
      res.json(postData);
    } catch (e) {
      res.status(500).json('ailed to post details');
    }
  });
});
app.post('/api/objective', async (req, res) => {
  mongoose.connect(process.env.MONGO_URL);
  const userData = await getUserDataFromToken(req);
  const { objective } = req.body;

  try {
    const postData = await Objective.create({
      objective,
      user: userData.id,
    });
    res.json(postData);
  } catch (e) {
    res.status(500).json('ailed to post details');
  }
});

// function to book a visit to resd
export const bookVisit = asyncHandler(async (req, res) => {
  const { email, date } = req.body;
  const { id } = req.params;

  try {
    const alreadyBooked = await prisma.user.findUnique({
      where: { email },
      select: { bookedVisits: true },
    });

    if (alreadyBooked.bookedVisits.some((visit) => visit.id === id)) {
      res
        .status(400)
        .json({ message: "This residency is already booked by you" });
    } else {
      await prisma.user.update({
        where: { email: email },
        data: {
          bookedVisits: { push: { id, date } },
        },
      });
      res.send("your visit is booked successfully");
    }
  } catch (err) {
    throw new Error(err.message);
  }
});

// funtion to get all bookings of a user
export const getAllBookings = asyncHandler(async (req, res) => {
  const { email } = req.body;
  try {
    const bookings = await prisma.user.findUnique({
      where: { email },
      select: { bookedVisits: true },
    });
    res.status(200).send(bookings);
  } catch (err) {
    throw new Error(err.message);
  }
});

// function to cancel the booking
export const cancelBooking = asyncHandler(async (req, res) => {
  const { email } = req.body;
  const { id } = req.params;
  try {
    const user = await prisma.user.findUnique({
      where: { email: email },
      select: { bookedVisits: true },
    });

    const index = user.bookedVisits.findIndex((visit) => visit.id === id);

    if (index === -1) {
      res.status(404).json({ message: "Booking not found" });
    } else {
      user.bookedVisits.splice(index, 1);
      await prisma.user.update({
        where: { email },
        data: {
          bookedVisits: user.bookedVisits,
        },
      });

      res.send("Booking cancelled successfully");
    }
  } catch (err) {
    throw new Error(err.message);
  }
});

// function to add a resd in favourite list of a user
export const toFav = asyncHandler(async (req, res) => {
  const { email } = req.body;
  const { rid } = req.params;

  try {
    const user = await prisma.user.findUnique({
      where: { email },
    });

    if (user.favResidenciesID.includes(rid)) {
      const updateUser = await prisma.user.update({
        where: { email },
        data: {
          favResidenciesID: {
            set: user.favResidenciesID.filter((id) => id !== rid),
          },
        },
      });

      res.send({ message: "Removed from favorites", user: updateUser });
    } else {
      const updateUser = await prisma.user.update({
        where: { email },
        data: {
          favResidenciesID: {
            push: rid,
          },
        },
      });
      res.send({ message: "Updated favorites", user: updateUser });
    }
  } catch (err) {
    throw new Error(err.message);
  }
});

// function to get all favorites
export const getAllFavorites = asyncHandler(async (req, res) => {
  const { email } = req.body;
  try {
    const favResd = await prisma.user.findUnique({
      where: { email },
      select: { favResidenciesID: true },
    });
    res.status(200).send(favResd);
  } catch (err) {
    throw new Error(err.message);
  }
});
