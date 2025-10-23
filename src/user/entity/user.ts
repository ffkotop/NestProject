import { Entity, PrimaryGeneratedColumn, Column, BaseEntity } from "typeorm"
import { IsEmpty, IsEmail, MinLength, MaxLength, Matches} from "class-validator"
@Entity()
export class User extends BaseEntity {
    @PrimaryGeneratedColumn()
    id: number

    @Column("varchar", {
        nullable : false,
    })
    @MinLength(4, {message : "имя должно содержать минимум 4 символа"})
    @MaxLength(100, {message : "имя не может содержать более чем 100 символов"})
    name: string

    @Column("varchar", {
        nullable : false
    })
    @MaxLength(150, { message : "почта не может содержать более чем 150 символов"})
    @MinLength(3, {message : "почта должна содержать минимум 3 символа"})
    @IsEmail()
    email : string

    @Column("varchar", {
        nullable : false, 

    })
    @MaxLength(255, { message : "Нельзя сделать пароль с более чем 255 символами"})
    @MinLength(8, { message: 'Пароль должен содержать минимум 8 символов' })
    @Matches(/^(?=.*[A-Z])(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]+$/, {
        message: 'Пароль должен содержать хотя бы одну заглавную букву и один специальный символ (!@#$%^&*)',
    })
    password : string
}