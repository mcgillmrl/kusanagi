// Inverted pendulum on cart controller
#include <TimerOne.h>
#include <TimerThree.h>
#include <Encoder.h>
#include <CmdMessenger.h>
#define BAUD_RATE 4000000
#define ENCODER_SAMPLING_PERIOD 1000 // microsecs
#define CONTROL_SAMPLING_PERIOD 1000 // microsecs
#define FORCE_TO_ANG_ACCEL 2048

#define DIR_PIN 7
#define PWM_PIN 6
#define PWM_RESOLUTION 13
//#define FOH_HOLD

// General params
double twopi = PI * 2;
elapsedMicros usec = 0;
unsigned long t_end = 0;
int PWM_MAX = 0.4*pow(2,PWM_RESOLUTION);
double IDEAL_PWM_FREQ = F_BUS/pow(2,PWM_RESOLUTION);
elapsedMicros encoder_usec = 0;
elapsedMicros control_usec = 0;
double enc_dt = 1e-6*ENCODER_SAMPLING_PERIOD;
double control_dt = 1e-6*CONTROL_SAMPLING_PERIOD;

// Encoder params
#define N_joints 2
Encoder* encoders[] = {new Encoder(14, 15),new Encoder(16, 17)};
double units_per_count[] = {0.32*twopi/30000, twopi/2000};
bool new_measurement=true;
union joint_states_type{
  // state is pos0,pos1,vel0,vel1,timestamp
  double as_double[5] = {0,0,0,0,0};
  char* as_byte[5*sizeof(double)];
};

joint_states_type joint_states;

// velocity estimator params
double encoder_gains[2][4] = {{35,250,400,10},{35,250,400,10}};
double joint_integral_error[2] = {0,0};
double vel_integral_error[2] = {0,0};
double joint_vels[2] = {0,0};
double joint_accels[2] = {0,0};
double previous_angles[2] = {0,0};
double w_slow[2] = {0,0};
double w_fast[2] = {0,0};
double tau[2] = {0.01,0.01};
double Kfast[2] = {10.0,10.0};
double a[2] = {0,0};
double b[2] = {0,0};
double current_angle,angle_error;

// current control signal
double u_t = 0;
double motor_vel = 0;
// angle limits
double angle_limits[2] = {-30000*units_per_count[0],30000*units_per_count[0]};
double min_angle_diff = 0;
double max_angle_diff = 0;
double limit_force = 0;

// serial command interface
CmdMessenger cmd = CmdMessenger(Serial);
// commands
enum
{
  // Commands
  RESET_STATE,
  GET_STATE,
  APPLY_CONTROL,
  CMD_OK,
  STATE
};

void setup() {
  // initialize serial port
  Serial.begin(BAUD_RATE);
  // setup PWM
  pinMode(DIR_PIN, OUTPUT);
  analogWriteResolution(PWM_RESOLUTION);
  analogWriteFrequency(PWM_PIN,IDEAL_PWM_FREQ);
  analogWrite(PWM_PIN,0);

  // initialize velocity estimator params
  for (int i=0; i<N_joints;i++){
    a[i] = 1/(tau[i] + 0.5*enc_dt);
    b[i] = a[i]*(tau[i] - 0.5*enc_dt);
  }
  
  // wait and initialize encoder loop
  Timer1.initialize(CONTROL_SAMPLING_PERIOD);
  Timer1.attachInterrupt(controlLoop);
  Timer3.initialize(ENCODER_SAMPLING_PERIOD);
  Timer3.attachInterrupt(readEncoders);
  
  // setup serial command interface
  cmd.printLfCr();
  cmd.attach(onUnknownCommand);
  cmd.attach(RESET_STATE,resetState);
  cmd.attach(GET_STATE,getCurrentState);
  cmd.attach(APPLY_CONTROL,applyControl);
  resetState();
  
}


void loop() {
  // read incoming commands
  cmd.feedinSerialData();
  // compute virtual force to avoid exceeding limit angles
  min_angle_diff = max(1e-6,10*(joint_states.as_double[0]-angle_limits[0]));
  max_angle_diff = min(-1e-6,10*(joint_states.as_double[0]-angle_limits[1]));
  limit_force = (1.0/(min_angle_diff*abs(min_angle_diff)) + 1.0/(max_angle_diff*abs(max_angle_diff)))*FORCE_TO_ANG_ACCEL;
  
  if (usec > t_end + 10000){
    u_t=0;
  }
  if (usec > t_end + 50000){
    motor_vel = 0;
  }
}

void onUnknownCommand(){
  cmd.sendCmd(0,"Command without attached callback");
}

void resetState(){
  //Serial.println(Serial.baud());
  usec=0;
  for (int i=0; i<N_joints;i++){
    encoders[i]->write(0);
    joint_integral_error[i]=0;
    joint_vels[i]=0;
    joint_accels[i]=0;
    previous_angles[i]=0;
    w_fast[i]=0;
    w_slow[i]=0;
    joint_states.as_double[i]=0;
    joint_states.as_double[i+N_joints]=0;
    encoder_usec=0;
    control_usec=0;
  }
  //cmd.sendCmd(CMD_OK,"Reset joint states to 0");
}

void getCurrentState(){
  cmd.sendCmdStart(STATE);
  for (int i=0; i<(2*N_joints+1);i++){
    cmd.sendCmdBinArg<double>(joint_states.as_double[i]);
  }
  cmd.sendCmdEnd();
}

void getCurrentStateError(){
  cmd.sendCmdStart(STATE);
  cmd.sendCmdBinArg<double>(units_per_count[1]*encoders[1]->read() - joint_states.as_double[1]);
  cmd.sendCmdBinArg<double>(0);
  cmd.sendCmdBinArg<double>(0);
  cmd.sendCmdBinArg<double>(0);
  cmd.sendCmdBinArg<double>(joint_states.as_double[2*N_joints]);
  cmd.sendCmdEnd();
}

void applyControl(){
#ifdef FOH_HOLD
  u_t = (cmd.readDoubleArg()*PWM_MAX - motor_vel);
  t_end = (unsigned long)(cmd.readDoubleArg()*1e6);
  u_t = u_t/((t_end - usec)*1e-6);
#else
  u_t = cmd.readDoubleArg()*PWM_MAX;
  t_end = (unsigned long)(cmd.readDoubleArg()*1e6);
#endif
}

void setMotorSpeed(double v){
  if (v>=0){
    v = min(PWM_MAX,v);
    digitalWrite(DIR_PIN,HIGH);
  }
  else{
    v = -max(-PWM_MAX,v);
    digitalWrite(DIR_PIN,LOW);
  }
  analogWrite(PWM_PIN,(int)v);
}

void readEncoders(){
  // compute dt
  enc_dt = 1e-6*encoder_usec;
  encoder_usec = 0;

  // update encoder estimates
  //PITrackingLoop();
  //PITrackingLoop_Accel();
  VelocityTrackingLoop();

  // fill in timestamp
  joint_states.as_double[2*N_joints] = usec*1e-6;  
}

void PITrackingLoop(){
  // tracking loop for speed estimate (see https://www.embeddedrelated.com/showarticle/530.php)
  for (int i=0; i<N_joints;i++){  
    joint_states.as_double[i] = units_per_count[i]*encoders[i]->read();
    double angle_error = joint_states.as_double[i] - previous_angles[i];
    joint_integral_error[i] += angle_error*encoder_gains[i][1]*enc_dt;
    joint_states.as_double[i+N_joints] = angle_error*encoder_gains[i][0] + joint_integral_error[i];
    previous_angles[i] += joint_states.as_double[i+N_joints]*enc_dt;
  }
}


void PITrackingLoop_Accel(){
  // tracking loop for speed estimate (see https://www.embeddedrelated.com/showarticle/530.php)
  for (int i=0; i<N_joints;i++){
    joint_states.as_double[i+N_joints] += joint_accels[i]*enc_dt;
    joint_states.as_double[i] += joint_states.as_double[i+N_joints]*enc_dt;
    
    current_angle = units_per_count[i]*encoders[i]->read();
    double current_vel = joint_vels[i];
    
    angle_error = current_angle - joint_states.as_double[i];
    double vel_error = current_vel - joint_states.as_double[i+N_joints];
    
    joint_integral_error[i] += angle_error*encoder_gains[i][1]*enc_dt;
    joint_vels[i] = angle_error*encoder_gains[i][0] + joint_integral_error[i];
    vel_integral_error[i] += vel_error*encoder_gains[i][3]*enc_dt;
    joint_accels[i] = vel_error*encoder_gains[i][2] + vel_integral_error[i];
  }
}


void VelocityTrackingLoop(){
  for (int i=0; i<N_joints;i++){
    current_angle = units_per_count[i]*encoders[i]->read();
    w_slow[i] = a[i]*(current_angle - previous_angles[i]) +b[i]*w_slow[i];
    w_fast[i] = 0;
    angle_error = current_angle - joint_states.as_double[i];
    if (abs(angle_error) >= 0.5*units_per_count[i]){
      w_fast[i] = angle_error*Kfast[i];
    }
    joint_states.as_double[i+N_joints] = w_slow[i] + w_fast[i];
    joint_integral_error[i] += joint_states.as_double[i+N_joints]*enc_dt;
    joint_states.as_double[i] = w_slow[i]*0.5*enc_dt + joint_integral_error[i];
    previous_angles[i] = current_angle;
  }
}

void controlLoop(){
  control_dt = 1e-6*control_usec;
  control_usec = 0;  
#ifdef FOH_HOLD  
  motor_vel += u_t*control_dt;
#else
  motor_vel = u_t;
#endif
  setMotorSpeed(motor_vel+limit_force*control_dt);
}

