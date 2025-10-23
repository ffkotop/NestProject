import { DataSource } from 'typeorm';
import { User } from './user/entity/user'; 

export const AppDataSource = new DataSource({
  type: 'postgres',
  host: process.env.DB_HOST || 'localhost',
  port: Number(process.env.DB_PORT) || 5432,
  username: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASS || 'm123456r',
  database: process.env.DB_NAME || 'user_database',
  entities: [User],
  migrations: ['src/migrations/*.ts'],
  synchronize: false,
});
