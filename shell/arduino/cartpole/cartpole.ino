// Inverted pendulum on cart controller
#include <TimerOne.h>
#include <TimerThree.h>
#include <Encoder.h>
#include <CmdMessenger.h>
#define BAUD_RATE 4000000
#define ENCODER_SAMPLING_PERIOD 128 // microsecs
#define ENC_DT 1e-6*ENCODER_SAMPLING_PERIOD
#define CONTROL_SAMPLING_PERIOD 128 // microsecs
#define CONTROL_DT 1e-6*CONTROL_SAMPLING_PERIOD
#define FORCE_TO_ANG_ACCEL 2048
#define VEL_SCALING 2048
//#define ACCEL_CONTROL

// General params
double twopi = PI * 2;
elapsedMicros usec = 0;
unsigned long t_end = 0;

// Encoder params
#define N_joints 2
Encoder* encoders[] = {new Encoder(0, 1),new Encoder(2, 3)};
double degrees_per_count[] = {twopi/30000, twopi/2000};
//double joint_angles[] = {0,0};
bool new_measurement=true;
union joint_states_type{
  // state is pos0,pos1,vel0,vel1,timestamp
  double as_double[5] = {0,0,0,0,0};
  char* as_byte[5*sizeof(double)];
};

joint_states_type joint_states;

// velocity estimator params
//double joint_angle_estimates[] = {0,0};
//double encoder_gains[2][2] = {{25,100},{25,100}};
double joint_integral_error[2] = {0,0};
double joint_angles[2] = {0,0};
double w_slow[2] = {0,0};
double w_fast[2] = {0,0};
double tau[2] = {0.008,0.008};
double Kfast[2] = {375,25};
double a[2] = {0,0};
double b[2] = {0,0};
double current_angle,angle_error;

// current control signal
double u_t = 0;
double motor_vel = 0;
// angle limits
double angle_limits[2] = {-2.5*PI,2.5*PI};
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
  //Serial.begin(BAUD_RATE);
  // setup DAC
  analogWriteResolution(12);
  analogWrite(A14, 2048);

  // initialize velocity estimator params
  for (int i=0; i<N_joints;i++){
    a[i] = 1/(tau[i] + 0.5*ENC_DT);
    b[i] = a[i]*(tau[i] - 0.5*ENC_DT);
  }
  
  // wait and initialize encoder loop
  Timer1.initialize(ENCODER_SAMPLING_PERIOD);
  Timer1.attachInterrupt(readEncoders);
  Timer3.initialize(CONTROL_SAMPLING_PERIOD);
  Timer3.attachInterrupt(controlLoop);

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
  limit_force = (1.0/(min_angle_diff*abs(min_angle_diff)) + 1.0/(max_angle_diff*abs(max_angle_diff)))*FORCE_TO_ANG_ACCEL*CONTROL_DT;
  
  if (usec > t_end + 50000){
    u_t=0;
    motor_vel = 0;
  }/* else {
    Serial.print(usec*1e-6,10);
    Serial.print(' ');
    Serial.print(limit_force,10);
    Serial.print(' ');
    Serial.println(motor_vel,10);
    delayMicroseconds(100);
  }
  */
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
    joint_angles[i]=0;
    w_fast[i]=0;
    w_slow[i]=0;
    joint_states.as_double[i]=0;
    joint_states.as_double[i+N_joints]=0;
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

void applyControl(){
#ifdef ACCEL_CONTROL
  u_t = cmd.readDoubleArg()*FORCE_TO_ANG_ACCEL*CONTROL_DT;
#else
  u_t = cmd.readDoubleArg()*VEL_SCALING;
#endif
  t_end = (unsigned long)(cmd.readDoubleArg()*1e6);
}

void setMotorSpeed(double v){
  v += 2048.0;
  v = min(4096,max(0,v));
  analogWrite(A14,(int)v);
}

void readEncoders(){  
  for (int i=0; i<N_joints;i++){
    /*
    joint_states.as_double[i] = degrees_per_count[i]*encoders[i]->read();
    // tracking loop for speed estimate (see https://www.embeddedrelated.com/showarticle/530.php)
    joint_angle_estimates[i] += joint_states.as_double[i+N_joints]*ENC_DT;
    double angle_error = joint_states.as_double[i] - joint_angle_estimates[i];
    if (abs(angle_error) > 1e-38){
      joint_integral_error[i] += angle_error*encoder_gains[i][1]*ENC_DT;
      joint_states.as_double[i+N_joints] = angle_error*encoder_gains[i][0] + joint_integral_error[i];
    }
    */
    current_angle = degrees_per_count[i]*encoders[i]->read();
    w_slow[i] = a[i]*(current_angle - joint_angles[i]) +b[i]*w_slow[i];
    w_fast[i] = 0;
    angle_error = current_angle - joint_states.as_double[i];
    if (abs(angle_error) >= degrees_per_count[i]){
      w_fast[i] = angle_error*Kfast[i];
    }
    
    joint_states.as_double[i+N_joints] = w_slow[i] + w_fast[i];
    joint_integral_error[i] += joint_states.as_double[i+N_joints]*ENC_DT;
    joint_states.as_double[i] = w_slow[i]*0.5*ENC_DT + joint_integral_error[i];
    joint_angles[i] = current_angle;
  }
  joint_states.as_double[2*N_joints] = usec*1e-6;
}

void controlLoop(){
#ifdef ACCEL_CONTROL  
  motor_vel += u_t;
  motor_vel = min(2048,max(-2048,motor_vel));
#else
  motor_vel = u_t;
#endif
  setMotorSpeed(motor_vel+limit_force);
}

