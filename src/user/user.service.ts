import { Injectable, ConflictException, UnauthorizedException } from '@nestjs/common';
import { User } from './entity/user'
import { Repository } from 'typeorm'
import { InjectRepository } from '@nestjs/typeorm';
import { CreateUserDto, LoginUserDto } from '../dto/user.dto'
import * as bcrypt from 'bcryptjs';
import { JwtService } from '@nestjs/jwt';
@Injectable()
export class UserService {
  constructor(
    @InjectRepository(User)
    private userRepository: Repository<User>,
    private jwtService: JwtService,
  ) {}

  async createUser(authInfo: CreateUserDto): Promise<any> {
    const email = authInfo.email.toLowerCase().trim(); 
    const existingUser = await this.findByEmail(email);
    if (existingUser) {
      throw new ConflictException('Пользователь с таким email уже существует');
    }
    const hashedPassword = await bcrypt.hash(authInfo.password, 10); 
    const newUser = this.userRepository.create({
      name: authInfo.name,
      email,
      password: hashedPassword,
    });
    const savedUser = await this.userRepository.save(newUser);
    const { password, ...result } = savedUser; 
    return result;
  }

  async findByEmail(email: string): Promise<User | null> {
    return await this.userRepository.findOne({ where: { email: email.toLowerCase().trim() } });
  }

  async verifyUser(authInfo: LoginUserDto) {
    const email = authInfo.email.toLowerCase().trim(); 
    const user = await this.findByEmail(email);
    console.log('Found user:', user); 
    if (user) {
      const isPasswordValid = await bcrypt.compare(authInfo.password, user.password);
      console.log('Password valid:', isPasswordValid, 'Stored hash:', user.password); 
      if (isPasswordValid) {
        const { password, ...result } = user;
        return result;
      }
    }
    throw new UnauthorizedException('Incorrect login or password');
  }

  async login(user: any) {
    const payload = { email: user.email, sub: user.id }; 
    return {
      access_token: this.jwtService.sign(payload),
    };
  }
}